import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from ModularGuide import ModularGuide
from guide_caption_search import GuideCaptionSearch


def _text_variants(question, answer, enabled):
    """Return the text views used by the Guide-side answer localizer."""
    variants = []
    for name in enabled:
        if name == "answer":
            variants.append(answer)
        elif name == "qa":
            variants.append((question + " " + answer).strip())
        elif name == "tail":
            variants.append(" ".join(answer.split()[-20:]))
        elif name == "last":
            parts = [part.strip() for part in answer.replace(" .", ".").split(". ")]
            parts = [part for part in parts if part]
            variants.append(parts[-1] if parts else answer)
    return variants


def load_path_rerank_weights(answer_model, ckpt_path):
    """Load a path-answer reranker checkpoint into a standalone LANA model."""
    try:
        checkpoint = torch.load(
            ckpt_path, map_location="cuda", weights_only=True)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cuda")
    if "vln_bert" not in checkpoint:
        raise ValueError(
            f"Reranker checkpoint {ckpt_path} has no vln_bert state")
    answer_model.agent.vln_bert.load_state_dict(
        checkpoint["vln_bert"], strict=False)
    if "critic" in checkpoint:
        answer_model.agent.critic.load_state_dict(
            checkpoint["critic"], strict=False)
    answer_model.agent.vln_bert.eval()
    answer_model.agent.critic.eval()
    return answer_model


class CompliantGuide(ModularGuide):
    def __init__(self, args, answer_model, localization_model, env_infos):
        super().__init__(args, answer_model, localization_model, env_infos)
        self.segment_steps = int(os.environ.get("GUIDE_SEGMENT_STEPS", "0"))
        self.goals_by_instr = env_infos["goals_by_instr"]
        self.targets_by_instr = env_infos.get("targets_by_instr", {})
        self.description_enabled = os.environ.get("GUIDE_DESC_ENABLED", "0") == "1"
        self.description_by_instr = self._load_descriptions(
            os.environ.get("GUIDE_DESCRIPTION_FILE", "")
        )
        # When the search is on, the guide works out its own wording in-episode
        # instead of reading a precomputed file.
        self.caption_search = GuideCaptionSearch(
            getattr(self, "shortest_distances", None)
        )
        self.arrival_confirm_enabled = (
            os.environ.get("GUIDE_ARRIVAL_CONFIRM", "0") == "1"
        )
        self.arrival_margin = float(
            os.environ.get("GUIDE_ARRIVAL_MARGIN", "0")
        )
        # Number of localization hypotheses consulted when deciding arrival.
        # 1 keeps the historical behaviour (argmax localization only); >1 also
        # accepts the runner-up hypotheses, which recovers arrivals that the
        # localizer ranks second because the panorama is ambiguous.
        self.arrival_topn = int(os.environ.get("GUIDE_ARRIVAL_TOPN", "1"))
        self.loc_consistency_enabled = (
            os.environ.get("GUIDE_LOC_CONSISTENCY", "0") == "1"
        )
        self.loc_history_limit = int(
            os.environ.get("GUIDE_LOC_HISTORY_LIMIT", "2")
        )
        self.loc_max_jump = float(
            os.environ.get("GUIDE_LOC_MAX_JUMP", "25")
        )
        self.localization_history = {}
        self.last_arrival_candidates = []
        self.last_arrival_judge_targets = []
        self.last_target_descriptions = []
        self.answer_grounding_models = []
        self.path_rerank_model = None

    def _load_descriptions(self, description_file):
        if not description_file:
            return {}
        try:
            payload = json.loads(Path(description_file).read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            records = payload["records"]
        elif isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            return {
                str(instr_id): str(description)
                for instr_id, description in payload.items()
                if description
            }
        else:
            return {}
        return {
            str(record["instr_id"]): str(
                record.get("description_text", record.get("description"))
            )
            for record in records
            if isinstance(record, dict)
            and record.get("instr_id") is not None
            and record.get("description_text", record.get("description"))
        }

    def get_goals(self, instr_ids):
        return [self.goals_by_instr[instr_id] for instr_id in instr_ids]

    def _localization_distance(self, scan, source, target):
        try:
            return float(
                self.localization_model.env.shortest_distances[scan][source][target]
            )
        except (AttributeError, KeyError, TypeError):
            return None

    def localize_consistent(
        self, scanIds, questions, instr_ids=None, active_indices=None
    ):
        instr_ids = instr_ids or list(range(len(questions)))
        active_indices = (
            set(range(len(questions)))
            if active_indices is None
            else set(active_indices)
        )
        localization_questions = list(questions)
        if self.loc_consistency_enabled:
            for index in active_indices:
                history = self.localization_history.get(str(instr_ids[index]), [])
                previous_questions = [
                    item["question"] for item in history[-self.loc_history_limit:]
                ]
                if previous_questions:
                    localization_questions[index] = " ".join(
                        previous_questions + [str(questions[index])]
                    )

        localized = self.localization_model.localize(
            scanIds, localization_questions
        )
        if not self.loc_consistency_enabled:
            return localized

        agent = getattr(self.localization_model, "agent", None)
        logits = getattr(agent, "last_loc_logits", None)
        nodes = getattr(agent, "last_loc_nodes", None)
        for index in active_indices:
            key = str(instr_ids[index])
            history = self.localization_history.setdefault(key, [])
            chosen = localized[index]
            if history and logits is not None and nodes is not None:
                previous = history[-1]["viewpoint"]
                candidate_nodes = nodes[index]
                candidate_scores = logits[index]
                allowed = []
                for node_index, node in enumerate(candidate_nodes):
                    distance = self._localization_distance(
                        scanIds[index], previous, node
                    )
                    if distance is not None and distance <= self.loc_max_jump:
                        allowed.append(node_index)
                if allowed and chosen not in {
                    candidate_nodes[node_index] for node_index in allowed
                }:
                    best_index = max(
                        allowed,
                        key=lambda node_index: float(candidate_scores[node_index]),
                    )
                    chosen = candidate_nodes[best_index]
            history.append(
                {"question": str(questions[index]), "viewpoint": chosen}
            )
            del history[:-self.loc_history_limit]
            localized[index] = chosen
        return localized

    def _answer_path(self, path):
        if self.segment_steps <= 0:
            return list(path)
        return list(path[:self.segment_steps + 1])

    def _localization_candidates(self, index, viewpoint):
        """Localization hypotheses for one batch entry, best score first."""
        if self.arrival_topn <= 1:
            return [viewpoint]
        agent = getattr(self.localization_model, "agent", None)
        nodes = getattr(agent, "last_loc_nodes", None)
        scores = getattr(agent, "last_loc_logits", None)
        candidates = [viewpoint]
        if nodes is None or scores is None or index >= len(nodes):
            return candidates
        try:
            entry_nodes = list(nodes[index])
            entry_scores = scores[index]
            ranked = sorted(
                range(min(len(entry_nodes), len(entry_scores))),
                key=lambda node_index: float(entry_scores[node_index]),
                reverse=True,
            )
        except (IndexError, TypeError, ValueError):
            return candidates
        for node_index in ranked[:self.arrival_topn]:
            node = entry_nodes[node_index]
            if node is not None and node not in candidates:
                candidates.append(node)
        return candidates

    def _arrival_judge_target(
        self,
        scan,
        goal,
        candidates,
        questions,
        instr_id,
    ):
        # Arrival is settled by the guide's own localizer.  The delivered
        # system offers no separate judge, so the shortlist is never overruled.
        return None

    def _at_goal(
        self,
        scan,
        viewpoint,
        goal,
        question=None,
        instr_id=None,
        candidates=None,
    ):
        if candidates:
            selected = self._arrival_judge_target(
                scan,
                goal,
                [item["viewpoint"] for item in candidates],
                [item.get("question", "") for item in candidates],
                instr_id,
            )
            if selected is not None:
                return selected == viewpoint
        if viewpoint in goal:
            return True
        if self.arrival_margin <= 0:
            return False
        return any(
            distance is not None and distance <= self.arrival_margin
            for target in goal
            for distance in [self._localization_distance(scan, viewpoint, target)]
        )

    def _decorate_answer(self, answer, description, at_goal):
        if at_goal and self.arrival_confirm_enabled and not description:
            return f"You have reached the target area. {answer}".strip()
        if not description:
            return answer
        lead = os.environ.get("GUIDE_CAPTION_LEADIN", "You should see")
        spoken = f"{lead} {description}." if lead else description
        if at_goal:
            prefix = "You have reached the target area."
            return f"{prefix} {spoken}".strip()
        return f"{answer.rstrip()} {spoken}".strip()

    def _localize_candidates_batch(
        self, model, scan, start_vp, texts, k, alpha
    ):
        """Get a merged top-k candidate set for several answer text views."""
        if model is None or not texts:
            return []
        agent = getattr(model, "agent", None)
        if agent is None:
            return []
        try:
            model.localize([scan] * len(texts), list(texts))
        except Exception as exc:
            if os.environ.get("GUIDE_PLAN_DEBUG") == "1":
                print(f"[GuidePlan] localization failed: {exc}")
            return []
        logits = getattr(agent, "last_loc_logits", None)
        nodes = getattr(agent, "last_loc_nodes", None)
        if logits is None or nodes is None or not nodes:
            return []

        try:
            dist_map = self.shortest_distances[scan][start_vp]
        except (KeyError, TypeError):
            return []
        merged = {}
        for batch_index in range(logits.shape[0]):
            probs = torch.softmax(logits[batch_index], dim=0)
            topk = torch.topk(probs, min(k, probs.shape[0]))
            node_list = nodes[batch_index]
            for node_index, value in zip(topk.indices, topk.values):
                candidate = node_list[int(node_index)]
                probability = float(value)
                if candidate == start_vp:
                    continue
                try:
                    path = list(
                        self.shortest_paths[scan][start_vp][candidate])
                except (KeyError, TypeError):
                    continue
                candidate_item = (
                    candidate,
                    probability,
                    dist_map.get(candidate, -1) + alpha * probability,
                    path,
                )
                previous = merged.get(candidate)
                if previous is None or probability > previous[1]:
                    merged[candidate] = candidate_item
        return sorted(merged.values(), key=lambda item: -item[1])[:k]

    def _union_candidates(self, models, scan, start_vp, texts, k, alpha):
        merged = {}
        for model in models:
            for candidate in self._localize_candidates_batch(
                model, scan, start_vp, texts, k, alpha):
                viewpoint, probability, _, path = candidate
                previous = merged.get(viewpoint)
                if previous is None or probability > previous[1]:
                    merged[viewpoint] = candidate
        return sorted(merged.values(), key=lambda item: -item[1])[:k]

    @torch.no_grad()
    def _score_paths(
        self,
        scan,
        paths,
        answer,
        question=None,
        batch_size=1,
        score_text_mode="answer",
    ):
        """Score Guide candidate paths by teacher-forced answer likelihood."""
        model = self.path_rerank_model or self.answer_model
        if model is None or not paths:
            return []
        agent = getattr(model, "agent", None)
        env = getattr(model, "language_env", None)
        if agent is None or env is None:
            return []

        if score_text_mode == "qa" and question:
            text = (question + " " + answer).strip()
        elif score_text_mode == "tail":
            text = " ".join(answer.split()[-20:])
        elif score_text_mode == "last":
            parts = [part.strip() for part in answer.replace(
                " .", ".").split(". ")]
            parts = [part for part in parts if part]
            text = parts[-1] if parts else answer
        else:
            text = answer

        try:
            token_ids = agent.tokenizer.encode(
                text, max_length=200, truncation=True)
        except TypeError:
            token_ids = agent.tokenizer.encode(text)[:200]
        if not token_ids:
            return []

        scores = []
        device = next(agent.vln_bert.parameters()).device
        for start in range(0, len(paths), max(1, batch_size)):
            chunk = paths[start:start + max(1, batch_size)]
            env.reset(
                [scan] * len(chunk),
                [path[0] for path in chunk],
                [3.14] * len(chunk),
                [],
            )
            t_hist, t_act, hist_lens, action_lens, _ = (
                agent.get_history_and_actions_for_speaker(env, chunk))
            batch_len = len(chunk)
            hist_embeds = [
                agent.vln_bert("history").expand(batch_len, -1, -1)
            ]
            action_embeds = []
            for t, action_input in enumerate(t_act):
                hist_embeds.append(agent.vln_bert(**t_hist[t]))
                action_embeds.append(
                    agent.vln_bert(**action_input).unsqueeze(1))

            words = torch.zeros(
                batch_len, len(token_ids), dtype=torch.long, device=device)
            token_tensor = torch.tensor(
                token_ids, dtype=torch.long, device=device)
            words[:, :len(token_ids)] = token_tensor
            future_mask = agent.make_future_mask(
                words.shape[1], hist_embeds[0].dtype, words.device)
            caption_lengths = (words != 0).sum(-1)
            ones = torch.ones_like(words)
            caption_mask = caption_lengths.unsqueeze(1) < ones.cumsum(dim=1)
            language_inputs = {
                "mode": "language",
                "txt_ids": words,
                "txt_masks": caption_mask,
                "future_mask": future_mask,
            }
            txt_embeds = agent.vln_bert(**language_inputs)
            caption_input = {
                "mode": "visual",
                "hist_embeds": hist_embeds,
                "txt_embeds": txt_embeds,
                "txt_masks": caption_mask,
                "hist_lens": hist_lens,
                "action_embeds": action_embeds,
                "action_lens": action_lens,
                "is_train_caption": True,
                "future_mask": future_mask,
            }
            logits = agent.vln_bert(**caption_input)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = words[:, 1:].contiguous()
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=agent.pad_token_id,
                reduction="none",
            ).view(batch_len, -1)
            valid = shift_labels != agent.pad_token_id
            scores.extend(
                (-ce.sum(dim=1) / valid.sum(dim=1).clamp(min=1))
                .detach().cpu().tolist()
            )
        return scores

    def plan_answer(
        self,
        scanIds,
        viewpoints,
        questions,
        goals,
        instr_ids=None,
        active_indices=None,
        arrival_viewpoints=None,
        confirm_indices=None,
        candidate_sets=None,
    ):
        active_indices = (
            list(range(len(scanIds)))
            if active_indices is None
            else list(active_indices)
        )
        answers = [""] * len(scanIds)
        seen_paths = [[] for _ in scanIds]
        plan_enabled = bool(self.answer_grounding_models or
                            self.path_rerank_model)
        if not plan_enabled:
            paths = [
                self._answer_path(
                    self._choose_path(scan, viewpoint, goal)
                )
                for scan, viewpoint, goal in zip(
                    scanIds, viewpoints, goals)
            ]
            if paths:
                active_answers, active_seen_paths = self.answer_model.answer(
                    scanIds,
                    viewpoints,
                    paths,
                )
                for index, answer, path in zip(
                    range(len(scanIds)), active_answers, active_seen_paths
                ):
                    answers[index] = answer
                    seen_paths[index] = path
        else:
            alpha = float(os.environ.get("GUIDE_PLAN_ALPHA", "5"))
            candidate_k = int(os.environ.get("GUIDE_PLAN_K", "20"))
            text_modes = [
                item.strip()
                for item in os.environ.get(
                    "GUIDE_PLAN_TEXTS", "answer,qa,tail,last").split(",")
                if item.strip()
            ]
            score_text_mode = os.environ.get(
                "GUIDE_PLAN_SCORE_TEXT", "answer")
            rerank_batch = int(
                os.environ.get("GUIDE_PLAN_RERANK_BATCH", "2"))
            use_rerank = os.environ.get(
                "GUIDE_PLAN_USE_RERANK", "1") == "1"
            for index in active_indices:
                scan = scanIds[index]
                viewpoint = viewpoints[index]
                question = questions[index]
                goal = goals[index]
                provisional_path = self._answer_path(
                    self._choose_path(scan, viewpoint, goal))
                provisional_answer, _ = self.answer_model.answer(
                    [scan], [viewpoint], [provisional_path])
                provisional_answer = provisional_answer[0]
                texts = _text_variants(
                    question, provisional_answer, text_modes)
                candidates = self._union_candidates(
                    self.answer_grounding_models,
                    scan,
                    viewpoint,
                    texts,
                    candidate_k,
                    alpha,
                )
                final_path = provisional_path
                if candidates and not use_rerank:
                    # Grounding-only ablation: use the highest-probability
                    # language-derived destination, without path reranking.
                    final_path = max(candidates, key=lambda item: item[1])[3]
                elif candidates and use_rerank:
                    candidate_paths = [item[3] for item in candidates]
                    if provisional_path not in candidate_paths:
                        # Keep the official Guide path as a no-regression
                        # option while letting the reranker override it when
                        # a language-derived candidate is better supported.
                        candidate_paths.append(provisional_path)
                    scores = self._score_paths(
                        scan,
                        candidate_paths,
                        provisional_answer,
                        question=question,
                        batch_size=rerank_batch,
                        score_text_mode=score_text_mode,
                    )
                    if scores:
                        final_path = candidate_paths[
                            max(range(len(scores)), key=lambda index: scores[index])
                        ]
                if final_path == provisional_path:
                    final_answer = provisional_answer
                else:
                    final_answer, _ = self.answer_model.answer(
                        [scan], [viewpoint], [final_path])
                    final_answer = final_answer[0]
                answers[index] = final_answer
                seen_paths[index] = final_path
                if os.environ.get("GUIDE_PLAN_DEBUG") == "1":
                    print(
                        f"[GuidePlan] candidates={len(candidates)} "
                        f"provisional_len={len(provisional_path)} "
                        f"final_len={len(final_path)}"
                    )

        instr_ids = instr_ids or [None] * len(answers)
        arrival_viewpoints = (
            list(viewpoints)
            if arrival_viewpoints is None
            else list(arrival_viewpoints)
        )
        confirm_indices = set(confirm_indices or [])
        decorated_answers = []
        self.last_arrival_candidates = []
        self.last_arrival_judge_targets = [None] * len(answers)
        self.last_target_descriptions = [""] * len(answers)
        for index, (scan, answer, instr_id, viewpoint, goal) in enumerate(
            zip(scanIds, answers, instr_ids, viewpoints, goals)
        ):
            if self.caption_search.enabled:
                description = self.caption_search.describe(
                    instr_id,
                    scan,
                    goal,
                    self.targets_by_instr.get(instr_id, ""),
                )
            else:
                description = (
                    self.description_by_instr.get(str(instr_id))
                    if self.description_enabled
                    else ""
                )
            self.last_target_descriptions[index] = str(description or "")
            arrival_candidates = self._localization_candidates(
                index, viewpoint
            )
            self.last_arrival_candidates.append(arrival_candidates)
            at_goal = False
            if not at_goal:
                at_goal = (
                    any(
                        self._at_goal(scan, hypothesis, goal)
                        for hypothesis in arrival_candidates
                    )
                    if self.arrival_confirm_enabled
                    else viewpoint in goal
                )
            decorated_answers.append(
                self._decorate_answer(answer, description, at_goal)
            )
        return decorated_answers, seen_paths
