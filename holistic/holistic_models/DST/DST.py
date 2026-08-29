import sys
import os
import re
from argparse import Namespace
from interface.WTA import WTA
from interface.Navigation import Navigation
import numpy as np
import torch
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
modules_path = os.path.join(current_dir, '../../../modules/nav/DST/map_nav_src')
sys.path.insert(0, modules_path)

try:
    from dst.dst import DST as NavDST
    from dst.env import NDHNavBatch
    from .default_args import get_default_args
    from utils.data import ImageFeaturesDB
    print("Successfully imported DST")
except ImportError as e:
    print(f"Import error: {e}")

from .clip_stop import ClipStopMatcher

def merge_args(default_args, new_args):
    """Merge new args with default args, only updating provided values"""
    if new_args is None:
        return default_args
    
    # Convert to dict for easier manipulation
    default_dict = vars(default_args)
    new_dict = vars(new_args) if hasattr(new_args, '__dict__') else new_args
    
    # Create merged dict
    merged_dict = default_dict.copy()
    
    # Only update values that are provided in new_args
    for key, value in new_dict.items():
        if value is not None:  # Only update if value is not None
            merged_dict[key] = value
    
    return Namespace(**merged_dict)

class DST(Navigation):
    def __init__(self, basepath, args=None, rank=0):
        from transformers import AutoTokenizer
        default_args = get_default_args(basepath)
        args = merge_args(default_args, args)
        
        super().__init__(args)
        self.rank = rank
        self.agent = NavDST(args, None, rank)
        # self.agent = GMapNavAgent(args, None, rank)
        self.feat_db = ImageFeaturesDB(self.args.val_ft_file, self.args.image_feat_size)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        if args.resume_file is not None:
            self.args.resume_iter  = self.agent.load(args.resume_file)
            print("Loaded the listener model at iter %d from %s" % (self.args.resume_iter, args.resume_file))

        ### navigation status
        self.obs = None
        self.gmaps = None
        self.instruction = []
        self.instruction_encoded = []
        self.language_inputs = None
        self.txt_embeds = None
        self.ended = None
        self.last_nav_inputs = None
        self.last_nav_vpids = None
        self.last_nav_logits = None
        self.last_nav_idx = None
        self.clip_stop_matcher = ClipStopMatcher()
        self.stop_texts = []
        self.arrival_answers = []
        self.confirm_arrival = np.array([], dtype=bool)
        self.confirm_ask_request = np.array([], dtype=bool)
        self.arrival_judge_targets = []
        self.target_descriptions = []
        self.frontier_states = []
        self.retro_clip_candidates = []
        self.visited_viewpoints = []
        self.forced_stop = np.array([], dtype=bool)
        self.dialog_suppressed = np.array([], dtype=bool)
    
    
    def eval(self):
        self.agent.vln_bert.eval()
        self.agent.critic.eval()

    def set_envs(self, envs, instr_data_dict):
        self.val_envs = {}
        for env in envs:
            if env not in instr_data_dict:
                raise ValueError(f"Environment {env} not found in instr_data_dict")
            navigator_data = [
                {
                    key: item[key]
                    for key in (
                        'instr_id', 'scan', 'start_pano', 'heading',
                        'instruction', 'instr_encoding', 'path_id')
                }
                for item in instr_data_dict[env]
            ]
            self.val_envs[env] = NDHNavBatch(self.feat_db,
                          navigator_data,
                          self.args.connectivity_dir,
                          batch_size=self.args.batch_size, 
                          angle_feat_size=self.args.angle_feat_size, 
                          seed=self.args.seed, name=env,
                          load_nav_graphs=False)
    
    def set_target_env(self, env_name):
        if env_name not in self.val_envs:
            raise ValueError(f"Environment {env_name} not found in val_envs")
        self.agent.env = self.val_envs[env_name]

    def reset_epoch(self):
        self.agent.env.reset_epoch(shuffle=False)
    
    def _get_state(self):
        return self.agent.env.env.getStates()
    
    def set_next_batch(self):
        kwargs = {'holistic': True}
        self.agent.env.reset(**kwargs)

    def get_obs(self):
        obs = []
        navigation_env = self.agent.env
        for i, (feature, state) in enumerate(self._get_state()):
            item = navigation_env.batch[i]
            base_view_id = state.viewIndex
           
            # Full features
            candidate = navigation_env.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)
            # [visual_feature, angle_feature] for views
            feature = np.concatenate((feature, navigation_env.angle_feature[base_view_id]), -1)

            ob = {
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'position': (state.location.x, state.location.y, state.location.z),
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'candidate': candidate,
                'navigableLocations' : state.navigableLocations,
                'instruction' : item['instruction'],
                'instr_encoding': item['instr_encoding'],
                # 'gt_path' : item['path'],
                'path_id' : item['path_id'],
            }
            # # RL reward. The negative distance between the state and the final state
            # # There are multiple gt end viewpoints on REVERIE. 
            # if ob['instr_id'] in self.gt_trajs:
            #     ob['distance'] = self.shortest_distances[ob['scan']][ob['viewpoint']][item['path'][-1]]
            # else:
            #     ob['distance'] = 0

            obs.append(ob)
        return obs

    def initialize_nav(self, obs):
        self.agent.feedback = 'argmax'
        self.gmaps = self.agent._initialize_graph(obs)
        self.ended = np.array([False] * len(obs))
        self.stop_texts = [""] * len(obs)
        self.arrival_answers = [False] * len(obs)
        self.confirm_arrival = np.array([False] * len(obs))
        self.confirm_ask_request = np.array([False] * len(obs))
        self.arrival_judge_targets = [None] * len(obs)
        self.target_descriptions = [""] * len(obs)
        self.visited_viewpoints = [[ob["viewpoint"]] for ob in obs]
        self.retro_clip_candidates = [[] for _ in obs]
        self.forced_stop = np.array([False] * len(obs))
        self.frontier_states = [
            {
                "history": [],
                "last_history_step": -1,
                "low_confidence": 0,
                "cooldown": 0,
                "forced": 0,
                "sweep_active": False,
                "sweep_done": False,
                "sweep_steps": 0,
                "stalled_steps": 0,
                "last_visited_count": 1,
                "retro_active": False,
                "retro_done": False,
                "retro_target": None,
                "retro_candidates": [],
                "retro_candidate_index": 0,
                "retro_confirm_pending": False,
                "retro_confirmations": 0,
                "asked_candidates": [],
                "retro_clip_candidates": [],
                "kway_asked": False,
            }
            for _ in obs
        ]
        self.dialog_suppressed = np.array([False] * len(obs))
        self.agent._update_graph_structure(obs, self.gmaps, self.ended)
        self.instruction = [ob['instruction'] for ob in obs]
        self.instruction_encoded = [self.tokenizer.encode(instr) for instr in self.instruction]
        with torch.no_grad():
            self.language_inputs, self.txt_embeds = self.agent._set_instruction(
                self.instruction_encoded
            )
        self.agent._update_scanvp_cands(obs)

    def set_arrival_judge_targets(self, targets):
        """Receive Guide's selected K-way destination for the next action."""
        if targets is None:
            return
        targets = list(targets)
        if len(self.arrival_judge_targets) != len(targets):
            self.arrival_judge_targets = [None] * len(targets)
        for index, target in enumerate(targets):
            if target is not None:
                self.arrival_judge_targets[index] = target

    def set_target_descriptions(self, descriptions):
        """Keep the Guide's spoken target description as a clean CLIP query.

        The text is already part of the answer the Navigator receives; this
        side channel only avoids re-parsing it out of the answer string.
        """
        if descriptions is None:
            return
        descriptions = list(descriptions)
        if len(self.target_descriptions) != len(descriptions):
            self.target_descriptions = [""] * len(descriptions)
        for index, description in enumerate(descriptions):
            if description:
                self.target_descriptions[index] = str(description)

    def get_confirm_candidate_sets(self):
        """CLIP-ranked visited nodes the guide should choose between."""
        return [
            list(state["retro_clip_candidates"])
            if state["retro_active"]
            else []
            for state in self.frontier_states
        ]

    def _clip_stop_logits(self, nav_logits, obs):
        if (
            not self.clip_stop_matcher.enabled
            or os.environ.get("CLIP_STOP_APPLY_LOGITS", "1") != "1"
        ):
            return nav_logits
        threshold = float(os.environ.get("CLIP_STOP_THRESHOLD", "0.30"))
        bias = float(os.environ.get("CLIP_STOP_BIAS", "0.80"))
        arrival_bias = float(
            os.environ.get("CLIP_STOP_ARRIVAL_BIAS", str(bias + 0.40))
        )
        adjusted_logits = nav_logits.clone()
        for index, ob in enumerate(obs):
            if self.ended[index] or not self.stop_texts[index]:
                continue
            score = self.clip_stop_matcher.score(
                ob["scan"],
                ob["viewpoint"],
                self.stop_texts[index],
            )
            if score is not None and score >= threshold:
                adjusted_logits[index, 0] += (
                    arrival_bias
                    if self.confirm_arrival[index]
                    else bias
                )
        return adjusted_logits

    def _select_frontier(self, index, ob):
        gmap = self.gmaps[index]
        current = ob["viewpoint"]
        candidates = []
        frontier_text = self._clip_query(index, ob)
        for viewpoint in gmap.node_positions:
            if viewpoint == current or gmap.graph.visited(viewpoint):
                continue
            try:
                route = gmap.graph.path(current, viewpoint)
                distance = gmap.graph.distance(current, viewpoint)
            except (KeyError, RecursionError):
                continue
            if not route:
                continue
            match = None
            if self.clip_stop_matcher.enabled and frontier_text:
                match = self.clip_stop_matcher.score(
                    ob["scan"],
                    viewpoint,
                    frontier_text,
                )
            candidates.append((match, distance, route[0]))
        if not candidates:
            return None
        if any(match is not None for match, _, _ in candidates):
            candidates.sort(
                key=lambda item: (
                    -(item[0] if item[0] is not None else -float("inf")),
                    item[1],
                )
            )
        else:
            candidates.sort(key=lambda item: item[1])
        return candidates[0][2]

    def _select_sweep_action(self, index, ob):
        gmap = self.gmaps[index]
        current = ob["viewpoint"]
        visited = set(self.visited_viewpoints[index])
        candidates = []
        for viewpoint in gmap.node_positions:
            if viewpoint == current or viewpoint in visited:
                continue
            try:
                route = gmap.graph.path(current, viewpoint)
                distance = gmap.graph.distance(current, viewpoint)
            except (KeyError, RecursionError):
                continue
            if not route:
                continue
            new_nodes = sum(node not in visited for node in route)
            candidates.append(
                (-new_nodes, distance, len(route), viewpoint, route[0])
            )
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][4]

    def _sweep_exploration(
        self, next_vp_ids, ended, nav_probs, obs, nav_idx
    ):
        start_step = int(os.environ.get("NAV_SWEEP_START", "8"))
        no_new_steps = max(
            1, int(os.environ.get("NAV_SWEEP_NO_NEW_STEPS", "5"))
        )
        repeat_window = max(
            2, int(os.environ.get("NAV_SWEEP_REPEAT_WINDOW", "4"))
        )
        max_repeat_nodes = max(
            1, int(os.environ.get("NAV_SWEEP_MAX_REPEAT_NODES", "2"))
        )
        max_sweep_steps = int(os.environ.get("NAV_SWEEP_MAX_STEPS", "0"))
        max_action_len = int(getattr(self.args, "max_action_len", 100))
        sweep_reserve = max(1, int(os.environ.get("NAV_SWEEP_RESERVE", "16")))
        if os.environ.get("NAV_CONFIRM_ASK", "0") == "1":
            # The candidate tour runs only after the sweep releases control, so
            # the sweep must hand over early enough to travel between candidates
            # and spend one dialog turn on each of them.
            sweep_reserve += max(
                0, int(os.environ.get("NAV_CONFIRM_RESERVE", "28"))
            )
        sweep_end_step = max(0, max_action_len - sweep_reserve)
        for index, ob in enumerate(obs):
            state = self.frontier_states[index]
            if self.ended[index]:
                continue
            if state["last_history_step"] != nav_idx:
                state["history"].append(ob["viewpoint"])
                state["history"] = state["history"][-repeat_window:]
                state["last_history_step"] = nav_idx
                # Counted per navigation step only: this method is also invoked
                # on dialog steps, which must not inflate the stall counter.
                visited_count = len(self.visited_viewpoints[index])
                if visited_count > state["last_visited_count"]:
                    state["stalled_steps"] = 0
                else:
                    state["stalled_steps"] += 1
                state["last_visited_count"] = visited_count

            if state["sweep_active"]:
                if (
                    nav_idx >= sweep_end_step
                    or (
                        max_sweep_steps > 0
                        and state["sweep_steps"] >= max_sweep_steps
                    )
                ):
                    state["sweep_active"] = False
                    state["sweep_done"] = True
                else:
                    sweep_action = self._select_sweep_action(index, ob)
                    if sweep_action is not None:
                        next_vp_ids[index] = sweep_action
                        ended[index] = False
                        state["sweep_steps"] += 1
                        continue
                    state["sweep_active"] = False
                    state["sweep_done"] = True

            if (
                state["sweep_done"]
                or nav_idx < start_step
                or nav_idx >= sweep_end_step
            ):
                continue
            looping = (
                len(state["history"]) >= repeat_window
                and len(set(state["history"][-repeat_window:]))
                <= max_repeat_nodes
            )
            stop_requested = bool(ended[index]) or next_vp_ids[index] is None
            if self.confirm_arrival[index]:
                # Only an arrival reported in reply to an explicit confirmation
                # request may skip the sweep; an arrival prefix produced by an
                # ordinary dialog turn is far too noisy to end the episode.
                stop_requested = False
            should_sweep = (
                stop_requested
                or state["stalled_steps"] >= no_new_steps
                or looping
            )
            if not should_sweep:
                continue
            state["sweep_active"] = True
            sweep_action = self._select_sweep_action(index, ob)
            if sweep_action is None:
                state["sweep_active"] = False
                state["sweep_done"] = True
                continue
            next_vp_ids[index] = sweep_action
            ended[index] = False
            state["sweep_steps"] += 1
        return next_vp_ids, ended

    def _frontier_exploration(self, next_vp_ids, ended, nav_probs, obs, nav_idx):
        if os.environ.get("NAV_SWEEP_ENABLED", "0") == "1":
            return self._sweep_exploration(
                next_vp_ids, ended, nav_probs, obs, nav_idx
            )
        if os.environ.get("NAV_FRONTIER_EXPLORE", "0") != "1":
            return next_vp_ids, ended
        min_step = int(os.environ.get("NAV_EXPLORE_MIN_STEP", "2"))
        confidence_threshold = float(
            os.environ.get("NAV_EXPLORE_CONFIDENCE", "0.35")
        )
        entropy_threshold = float(
            os.environ.get("NAV_EXPLORE_ENTROPY", "2.0")
        )
        confidence_streak = int(
            os.environ.get("NAV_EXPLORE_STREAK", "2")
        )
        max_forced = int(os.environ.get("NAV_EXPLORE_MAX_STEPS", "3"))
        repeat_window = int(os.environ.get("NAV_EXPLORE_REPEAT_WINDOW", "4"))
        max_repeat_nodes = int(os.environ.get("NAV_EXPLORE_MAX_REPEAT_NODES", "2"))
        for index, ob in enumerate(obs):
            state = self.frontier_states[index]
            if self.ended[index]:
                continue
            viewpoint = ob["viewpoint"]
            if state["last_history_step"] != nav_idx:
                state["history"].append(viewpoint)
                state["history"] = state["history"][-repeat_window:]
                state["last_history_step"] = nav_idx
            probabilities = nav_probs[index]
            confidence = float(probabilities.max().item())
            entropy = float(
                -(probabilities * probabilities.clamp_min(1e-8).log()).sum().item()
            )
            if confidence < confidence_threshold or entropy > entropy_threshold:
                state["low_confidence"] += 1
            else:
                state["low_confidence"] = 0
            looping = (
                len(state["history"]) >= repeat_window
                and len(set(state["history"])) <= max_repeat_nodes
            )
            should_explore = (
                nav_idx >= min_step
                and state["cooldown"] == 0
                and state["forced"] < max_forced
                and (
                    state["low_confidence"] >= confidence_streak
                    or looping
                )
            )
            if should_explore:
                frontier_action = self._select_frontier(index, ob)
                if frontier_action is not None:
                    next_vp_ids[index] = frontier_action
                    ended[index] = False
                    state["forced"] += 1
                    state["cooldown"] = 1
                    continue
            if state["cooldown"] > 0:
                state["cooldown"] -= 1
        return next_vp_ids, ended

    ROOM_NAMES = (
        "living room",
        "dining room",
        "family room",
        "utility room",
        "bedroom",
        "bathroom",
        "kitchen",
        "hallway",
        "entryway",
        "office",
        "stairway",
        "garage",
        "patio",
        "washroom",
        "loft",
        "attic",
        "basement",
    )

    # Object categories the navigator can recognise in a guide answer.  CLIP
    # ranks a bare noun list better than the guide's full sentence, whose
    # imperative wrapper ("look for ...", "you should also see ...") spends the
    # text budget on function words.  Same technique as ROOM_NAMES above, only
    # widened to objects; nothing here is episode specific.
    OBJECT_NAMES = (
        "arch", "archway", "armchair", "bathtub", "bed", "bench", "blinds",
        "books", "bookshelf", "cabinet", "chair", "clock", "closet", "column",
        "couch", "counter", "countertop", "curtain", "desk", "dishwasher",
        "dresser", "faucet", "fence", "fireplace", "footstool", "fountain",
        "handrail", "headboard", "lamp", "mat", "microwave", "mirror",
        "nightstand", "painting", "phone", "pillow", "plant", "planter",
        "plants", "plug", "rack", "railing", "seat", "shelf", "shelves",
        "sink", "skylight", "stair", "stairs", "stool", "stove", "table",
        "thermostat", "toilet", "towel", "towels", "trashcan", "tv", "vanity",
        "vent", "wardrobe", "washbasin", "windowframe",
    )

    def _mentioned_room(self, text):
        lowered = text.lower()
        mentions = [
            (lowered.rfind(room), room)
            for room in self.ROOM_NAMES
            if re.search(
                r"(?<![a-z])" + re.escape(room) + r"(?![a-z])", lowered
            )
        ]
        return max(mentions)[1] if mentions else ""

    def _mentioned_objects(self, text, limit=4):
        lowered = text.lower()
        found = []
        for name in sorted(self.OBJECT_NAMES, key=len, reverse=True):
            if len(found) >= limit:
                break
            if any(name in existing for existing in found):
                continue
            if re.search(
                r"(?<![a-z])" + re.escape(name) + r"s?(?![a-z])", lowered
            ):
                found.append(name)
        return found

    def _clip_query(self, index, ob):
        answer_text = self.stop_texts[index]
        if not answer_text:
            return ""
        target_text = str(ob.get("instruction", "")).strip()
        if ":" in target_text:
            target_text = target_text.split(":", 1)[1].strip()
        query_mode = os.environ.get("NAV_RETRO_CLIP_TEXT", "template")
        description = (
            self.target_descriptions[index]
            if index < len(self.target_descriptions)
            else ""
        )
        if query_mode == "target_desc" and description:
            return f"a photo of a {target_text}. {description}"
        if query_mode == "desc" and description:
            return description
        if query_mode == "answer":
            return answer_text
        if query_mode == "target_answer":
            return f"a photo of a {target_text}. {answer_text}"
        if query_mode == "answer_desc":
            # The guide speaks the words after a fixed lead-in, so the dialog
            # text is the only path they travel; take that span verbatim.
            marker = os.environ.get(
                "NAV_DESC_MARKER", "you should see"
            ).lower()
            lowered = answer_text.lower()
            position = lowered.rfind(marker) if marker else -1
            if position >= 0:
                spoken = answer_text[position + len(marker):]
                spoken = spoken.strip().rstrip(".").strip()
                if spoken:
                    return spoken
            return answer_text
        if query_mode in {"vocab_bag", "vocab_target_objs", "answer_bag"}:
            # answer_bag ignores the side channel and reads the words back out
            # of the guide's spoken answer, so the dialog is the only path the
            # description travels.
            source = (
                answer_text
                if query_mode == "answer_bag"
                else (description or answer_text)
            )
            room = self._mentioned_room(source)
            objects = [
                name
                for name in self._mentioned_objects(source)
                if name != target_text and name != room
            ]
            if query_mode in {"vocab_bag", "answer_bag"}:
                parts = [
                    part
                    for part in ([target_text, room] + objects)
                    if part
                ]
                if parts:
                    return ", ".join(parts)
            elif objects:
                joined = ", ".join(objects)
                return (
                    f"a photo of a {target_text} in a {room} with {joined}"
                    if room
                    else f"a photo of a {target_text} with {joined}"
                )
            elif room:
                return f"a photo of a {target_text} in a {room}"
            return f"a photo of a {target_text}"
        room = self._mentioned_room(answer_text)
        return (
            f"a photo of a {target_text} in a {room}"
            if room
            else f"a photo of a {target_text}. {answer_text}"
        )

    def _select_retro_candidates(self, index, ob):
        self.retro_clip_candidates[index] = []
        if (
            os.environ.get("NAV_RETRO_CLIP", "0") == "1"
            and self.clip_stop_matcher.enabled
            and self.stop_texts[index]
        ):
            clip_text = self._clip_query(index, ob)
            top_k = int(os.environ.get("NAV_RETRO_CLIP_TOPK", "12"))
            candidates = []
            for viewpoint in self.visited_viewpoints[index]:
                score = self.clip_stop_matcher.score_topk(
                    ob["scan"],
                    viewpoint,
                    clip_text,
                    top_k=top_k,
                )
                if score is not None:
                    candidates.append((score, viewpoint))
            if candidates:
                candidates.sort(key=lambda item: (-item[0], item[1]))
                threshold = float(
                    os.environ.get("NAV_RETRO_CLIP_THRESHOLD", "0.245")
                )
                candidate_count = max(
                    1,
                    int(os.environ.get("NAV_CONFIRM_TOPK", "3")),
                )
                clip_ranked = [
                    viewpoint
                    for _, viewpoint in candidates[:candidate_count]
                ]
                self.retro_clip_candidates[index] = list(clip_ranked)
                if candidates[0][0] >= threshold:
                    return clip_ranked
                # CLIP is not confident enough to lead, so the navigator's own
                # stop score takes the first slot, but the CLIP candidates are
                # still worth touring instead of collapsing the list to one node.
                merged = self._stop_score_candidates(index) + clip_ranked
                deduped = list(dict.fromkeys(merged))[:candidate_count]
                return deduped or [ob["viewpoint"]]

        return self._stop_score_candidates(index) or [ob["viewpoint"]]

    def _stop_score_candidates(self, index):
        """Best visited viewpoint by the navigator's own stop score."""
        candidates = []
        for viewpoint, score in self.gmaps[index].node_stop_scores.items():
            if viewpoint in self.visited_viewpoints[index]:
                candidates.append(
                    (float(score.get("stop", -float("inf"))), viewpoint)
                )
        if not candidates:
            return []
        return [max(candidates, key=lambda item: (item[0], item[1]))[1]]

    def _select_retro_target(self, index, ob):
        return self._select_retro_candidates(index, ob)[0]

    def _retro_route(self, index, current, target, remaining):
        """Route to target, or [] when it does not fit the remaining budget."""
        if target is None or target == current:
            return []
        try:
            route = self.gmaps[index].graph.path(current, target)
        except (KeyError, RecursionError):
            return []
        if not route or len(route) >= remaining:
            return []
        return route

    def _retro_fallback_target(self, state, current):
        """Best already-asked candidate to fall back to once the tour ends."""
        asked = set(state["asked_candidates"])
        # Prefer CLIP order even when the threshold-bypass path puts the
        # navigator's stop-score candidate at the head of the tour.
        for viewpoint in state["retro_clip_candidates"]:
            if viewpoint in asked:
                return viewpoint
        for viewpoint in state["asked_candidates"]:
            return viewpoint
        if state["retro_candidates"]:
            return state["retro_candidates"][0]
        return current

    def _retro_stop_action(self, next_vp_ids, ended, obs, nav_idx):
        if os.environ.get("NAV_RETRO_STOP", "0") != "1":
            return next_vp_ids, ended
        min_step = int(os.environ.get("NAV_RETRO_MIN_STEP", "8"))
        reserve = max(0, int(os.environ.get("NAV_RETRO_RESERVE", "8")))
        max_action_len = int(getattr(self.args, "max_action_len", 100))
        confirm_enabled = os.environ.get("NAV_CONFIRM_ASK", "0") == "1"
        confirm_limit = max(
            1,
            int(os.environ.get("NAV_CONFIRM_TOPK", "3")),
        )
        kway_mode = os.environ.get("NAV_CONFIRM_KWAY", "0")
        kway_enabled = kway_mode in {"1", "oneshot"}
        kway_oneshot = kway_mode == "oneshot"
        for index, ob in enumerate(obs):
            state = self.frontier_states[index]
            if self.ended[index] or state["retro_done"]:
                continue
            judge_target = (
                self.arrival_judge_targets[index]
                if index < len(self.arrival_judge_targets)
                else None
            )
            if kway_enabled and judge_target is not None:
                current = ob["viewpoint"]
                state["retro_confirm_pending"] = False
                self.confirm_ask_request[index] = False
                state["retro_target"] = judge_target
                state["retro_active"] = True
                if current == judge_target:
                    next_vp_ids[index] = current
                    ended[index] = True
                    state["retro_active"] = False
                    state["retro_done"] = True
                    self.arrival_judge_targets[index] = None
                    self.forced_stop[index] = True
                    continue
                remaining = max_action_len - nav_idx
                route = self._retro_route(
                    index,
                    current,
                    judge_target,
                    remaining,
                )
                if route:
                    next_vp_ids[index] = route[0]
                    ended[index] = False
                    continue
                next_vp_ids[index] = current
                ended[index] = True
                state["retro_active"] = False
                state["retro_done"] = True
                self.arrival_judge_targets[index] = None
                self.forced_stop[index] = True
                continue
            if not state["retro_active"]:
                if (
                    self.confirm_arrival[index]
                    and not state["sweep_done"]
                ):
                    continue
                stop_requested = bool(ended[index]) or next_vp_ids[index] is None
                near_budget = nav_idx >= max_action_len - reserve
                if not state["sweep_done"] and (
                    nav_idx < min_step or not (stop_requested or near_budget)
                ):
                    continue
                state["retro_candidates"] = self._select_retro_candidates(
                    index,
                    ob,
                )
                state["retro_candidate_index"] = 0
                state["retro_confirm_pending"] = False
                state["retro_confirmations"] = 0
                state["asked_candidates"] = []
                state["retro_clip_candidates"] = list(
                    self.retro_clip_candidates[index]
                )
                state["retro_target"] = state["retro_candidates"][0]
                state["retro_active"] = True

            if (
                kway_oneshot
                and not state["kway_asked"]
                and state["retro_clip_candidates"]
            ):
                # Hand the whole CLIP top-K set to the guide in a single turn so
                # the referee costs one dialog instead of one per candidate.
                state["kway_asked"] = True
                state["retro_confirm_pending"] = False
                self.arrival_answers[index] = False
                self.confirm_arrival[index] = False
                self.confirm_ask_request[index] = True
                next_vp_ids[index] = ob["viewpoint"]
                ended[index] = False
                continue
            if kway_oneshot and state["kway_asked"]:
                # The referee already had its turn; a missing answer must not
                # trigger a second ask, so fall back to the CLIP-led tour.
                self.confirm_ask_request[index] = False

            if state["retro_confirm_pending"]:
                if self.arrival_answers[index] and not kway_enabled:
                    current = ob["viewpoint"]
                    next_vp_ids[index] = current
                    ended[index] = True
                    state["retro_active"] = False
                    state["retro_done"] = True
                    state["retro_confirm_pending"] = False
                    self.confirm_ask_request[index] = False
                    self.forced_stop[index] = True
                    continue
                state["retro_confirm_pending"] = False
                self.confirm_ask_request[index] = False
                state["retro_candidate_index"] += 1
                if (
                    state["retro_candidate_index"]
                    >= len(state["retro_candidates"])
                    or state["retro_confirmations"] >= confirm_limit
                ):
                    state["retro_target"] = self._retro_fallback_target(
                        state,
                        ob["viewpoint"],
                    )
                else:
                    state["retro_target"] = state["retro_candidates"][
                        state["retro_candidate_index"]
                    ]

            target = state["retro_target"]
            current = ob["viewpoint"]
            if target is None or target == current:
                candidate_index = state["retro_candidate_index"]
                if (
                    confirm_enabled
                    and candidate_index < len(state["retro_candidates"])
                    and state["retro_confirmations"] < confirm_limit
                    and target == state["retro_candidates"][candidate_index]
                ):
                    self.arrival_answers[index] = False
                    self.confirm_arrival[index] = False
                    state["retro_confirm_pending"] = True
                    state["retro_confirmations"] += 1
                    if current not in state["asked_candidates"]:
                        state["asked_candidates"].append(current)
                    self.confirm_ask_request[index] = True
                    next_vp_ids[index] = current
                    ended[index] = False
                    continue
                next_vp_ids[index] = current
                ended[index] = True
                state["retro_active"] = False
                state["retro_done"] = True
                self.confirm_ask_request[index] = False
                self.forced_stop[index] = True
                continue
            remaining = max_action_len - nav_idx
            route = self._retro_route(index, current, target, remaining)
            if not route and confirm_enabled:
                # This candidate does not fit the remaining budget, but a later
                # one may still be close enough to visit and ask about.
                if state["retro_confirmations"] < confirm_limit:
                    for offset in range(
                        state["retro_candidate_index"] + 1,
                        len(state["retro_candidates"]),
                    ):
                        candidate = state["retro_candidates"][offset]
                        route = self._retro_route(
                            index, current, candidate, remaining
                        )
                        if route:
                            state["retro_candidate_index"] = offset
                            state["retro_target"] = candidate
                            break
            if not route:
                # No candidate is left to visit. Stopping where we stand would
                # end the episode on a node the guide already rejected, so walk
                # back to the best candidate we did ask about instead.
                fallback = self._retro_fallback_target(state, current)
                route = self._retro_route(index, current, fallback, remaining)
                state["retro_target"] = fallback if route else current
            if route:
                next_vp_ids[index] = route[0]
                ended[index] = False
            else:
                next_vp_ids[index] = current
                ended[index] = True
                state["retro_active"] = False
                state["retro_done"] = True
                self.confirm_ask_request[index] = False
                self.forced_stop[index] = True
        return next_vp_ids, ended

    def get_next_action(self, nav_idx, obs):
        with torch.no_grad():
            nav_inputs, pano_inputs = self.agent._process_navigation_step(
                obs,
                self.gmaps,
                self.ended,
                nav_idx,
                self.language_inputs,
                self.txt_embeds,
            )
            nav_probs, nav_vpids, nav_logits, nav_outs = self.agent._nav_probs(
                nav_inputs
            )
            nav_logits = self._clip_stop_logits(nav_logits, obs)
            nav_probs = torch.softmax(nav_logits, dim=1)
            self.agent._update_stop_scores(nav_probs, self.gmaps, obs, self.ended)
        self.last_nav_inputs = nav_inputs
        self.last_nav_vpids = nav_vpids
        self.last_nav_logits = nav_logits
        self.last_nav_idx = nav_idx

        # decide next action. feedback = argmax
        _, _, next_vp_ids, ended, _ = self.agent.decide_action(nav_logits, nav_probs, nav_vpids, nav_inputs, obs, self.ended, len(obs), nav_idx)
        next_vp_ids, ended = self._frontier_exploration(
            next_vp_ids,
            ended,
            nav_probs,
            obs,
            nav_idx,
        )
        next_vp_ids, ended = self._retro_stop_action(
            next_vp_ids,
            ended,
            obs,
            nav_idx,
        )
        self.dialog_suppressed = np.array(
            [
                state["sweep_active"] or state["retro_active"]
                for state in self.frontier_states
            ],
            dtype=bool,
        )
        instrucion_for_this_nav = [self.tokenizer.decode(seq[seq.nonzero().squeeze()]) for seq in self.language_inputs['txt_ids']]
        return next_vp_ids, ended, nav_probs, instrucion_for_this_nav, nav_outs

    def apply_local_grounding(self, indices, local_logits, local_nodes, obs,
                              weight=0.4):
        if self.last_nav_logits is None or self.last_nav_vpids is None:
            raise RuntimeError("navigation scores are not initialized")
        if not indices:
            return (
                self.agent.decide_action(
                    self.last_nav_logits, torch.softmax(
                        self.last_nav_logits, dim=1), self.last_nav_vpids,
                        self.last_nav_inputs, obs, self.ended, len(obs),
                        self.last_nav_idx)[2],
                self.ended.copy(),
                torch.softmax(self.last_nav_logits, dim=1),
            )

        adjusted_logits = self.last_nav_logits.clone()
        local_probs = torch.softmax(local_logits, dim=1).cpu()
        for local_row, batch_idx in enumerate(indices):
            if batch_idx >= len(self.last_nav_vpids):
                continue
            node_scores = {
                node: float(local_probs[local_row, node_idx])
                for node_idx, node in enumerate(local_nodes[local_row])
                if node_idx < local_probs.shape[1]
            }
            for action_idx, viewpoint in enumerate(self.last_nav_vpids[batch_idx]):
                if viewpoint is None or viewpoint not in node_scores:
                    continue
                adjusted_logits[batch_idx, action_idx] += weight * np.log(
                    max(node_scores[viewpoint], 1e-8))

        adjusted_probs = torch.softmax(adjusted_logits, dim=1)
        self.agent._update_stop_scores(
            adjusted_probs, self.gmaps, obs, self.ended)
        _, _, next_vp_ids, ended, _ = self.agent.decide_action(
            adjusted_logits, adjusted_probs, self.last_nav_vpids,
            self.last_nav_inputs, obs, self.ended, len(obs),
            self.last_nav_idx)
        return next_vp_ids, ended, adjusted_probs
    
    def navigate(self, next_vp_ids, obs, just_ended, traj):
        previous_vp_ids = [ob['viewpoint'] for ob in obs]
        actions = list(next_vp_ids)
        for index, forced_stop in enumerate(self.forced_stop):
            if forced_stop:
                actions[index] = None
        self.agent.make_equiv_action(actions, self.gmaps, obs, traj)
        new_obs = self.get_obs()
        update_stop = just_ended & ~self.forced_stop
        self.agent._update_stop_node(
            new_obs,
            self.gmaps,
            self.ended,
            update_stop,
            traj,
            len(obs),
        )
        self.agent._update_graph_structure(new_obs, self.gmaps, self.ended)
        # self.ended[:] = np.logical_or(self.ended, np.array([x is None for x in next_vp_ids]))
        self.ended[:] = np.logical_or(self.ended, just_ended)

        for index, ob in enumerate(new_obs):
            viewpoint = ob["viewpoint"]
            if viewpoint not in self.visited_viewpoints[index]:
                self.visited_viewpoints[index].append(viewpoint)
        paths = []
        for index in range(len(obs)):
            if next_vp_ids[index] is None:
                paths.append([])
                continue
            try:
                paths.append(
                    self.gmaps[index].graph.path(
                        previous_vp_ids[index],
                        next_vp_ids[index],
                    )
                )
            except (KeyError, RecursionError):
                paths.append([])
        for index, path in enumerate(paths):
            for viewpoint in path:
                if viewpoint not in self.visited_viewpoints[index]:
                    self.visited_viewpoints[index].append(viewpoint)
        self.forced_stop[:] = False
        return new_obs, paths

    def update_instruction(self, to_ask_indices, questions, answers, append_behind=False):
        obs = self.get_obs()
        new_instructions = []
        new_instructions_encoded = []
        for i, ob in enumerate(obs):
            if i in to_ask_indices:
                self.stop_texts[i] = str(answers[i])
                self.arrival_answers[i] = str(answers[i]).lstrip().lower().startswith(
                    "you have reached the target area"
                )
                # Remember whether this arrival report answers an explicit
                # confirmation request; the flag is still set at this point
                # because the retro state machine clears it on the next step.
                self.confirm_arrival[i] = bool(
                    self.arrival_answers[i]
                ) and bool(self.confirm_ask_request[i])
            target_encoded = self.tokenizer.encode(ob["instruction"])
            answer_encoded = self.tokenizer.encode(answers[i])
            if append_behind: # for R2R, RxR test
                new_instructions.append(f"{ob['instruction']} {answers[i]}")
                encoded = target_encoded + answer_encoded[1:]
            else:
                new_instructions.append(f"{answers[i]} {ob['instruction']}")
                encoded = answer_encoded + target_encoded[1:]
            new_instructions_encoded.append(encoded[:512])

        for i in to_ask_indices:
            self.instruction[i] = new_instructions[i]
            self.instruction_encoded[i] = new_instructions_encoded[i]

        with torch.no_grad():
            self.language_inputs, self.txt_embeds = self.agent._set_instruction(
                self.instruction_encoded
            )

    
    # def wta(self, step, nav_probs, nav_outs):
    #     ask = self.agent.decide_wta(nav_outs)
    #     return ask