from parser import parse_args
import torch
from holistic_utils.distributed import init_distributed, is_default_gpu
from holistic_utils.misc import set_random_seed
from holistic_utils.data_utils import construct_instrs, construct_instrs_universal
from holistic_models.ScaleVLN.ScaleVLN import ScaleVLNModel
from holistic_models.DST.DST import DST
from ModularNavigator import ModularNavigator
from CompliantGuide import CompliantGuide, load_path_rerank_weights
from holistic_models.FixedInterval import FixedIntervalWtaModule
from holistic_models.ConfidenceThresholding import ConfidenceThresholdingWtaModule
from holistic_models.HybridWTA import HybridWtaModule, CappedConfidenceWtaModule
from holistic_models.FixedResponse import FixedAnswerGeneration, FixedQuestionGeneration
from holistic_models.LANA.LANA import LANA
from holistic_models.GCNLoc.GCNLoc import GCNLocModel
from holistic_models.GTL.GTL import GraphVlnAgentModel
import time
import numpy as np
from evaluator import Evaluator
import copy
import os
import json
from transformers import logging
import sys
logging.set_verbosity_error()


def _configure_cpu_threads():
    """Optionally cap Torch CPU pools for multi-container evaluation."""
    value = os.environ.get("DIALNAV_TORCH_THREADS")
    if not value:
        return
    try:
        threads = max(1, int(value))
    except ValueError as exc:
        raise ValueError(
            f"DIALNAV_TORCH_THREADS must be an integer, got {value!r}"
        ) from exc
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        # A caller embedding main.py may have initialized the inter-op pool
        # already; the intra-op cap is still useful in that case.
        pass
    print(f"Configured Torch CPU threads: intra/inter={threads}")


def _profile_add(profile, key, value=0.0):
    if profile is not None:
        profile[key] = profile.get(key, 0.0) + value


def get_tokenizer():
    from transformers import AutoTokenizer
    cfg_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(cfg_name)
    return tokenizer

def load_instruction_data(args, target_envs, tokenizer):
    env_instructions = {}
    for split in target_envs:
        if split == "val_seen":
            annotation_paths = args.val_seen_anno_paths
        elif split == "val_unseen":
            annotation_paths = args.val_unseen_anno_paths
        elif split == "test":
            annotation_paths = args.test_anno_paths
        else:
            raise ValueError(f"Invalid split: {split}")
        
        annotation_paths = annotation_paths.split(",")
        instruction_data = construct_instrs(annotation_paths, tokenizer, args.max_instr_len)
        print(f"Loaded instruction data for split: {split} ({annotation_paths}) with length: {len(instruction_data)}")
        env_instructions[split] = instruction_data
    return env_instructions

def load_instruction_data_universal(args, target_envs, tokenizer):
    env_instructions = {}

    if args.benchmark in ['cvdn', 'dialnav']:
        prefix = "target : "
    else:
        prefix = ""

    for split in target_envs:
        if split == "val_seen":
            annotation_paths = args.val_seen_anno_paths
        elif split == "val_unseen":
            annotation_paths = args.val_unseen_anno_paths
        elif split == "test":
            annotation_paths = args.test_anno_paths
        else:
            raise ValueError(f"Invalid split: {split}")

        annotation_paths = annotation_paths.split(",")
        instruction_data = construct_instrs_universal(annotation_paths, tokenizer, args.max_instr_len, prefix)
        print(f"Loaded instruction data for split: {split} ({annotation_paths}) with length: {len(instruction_data)}")
        env_instructions[split] = instruction_data
    return env_instructions

def dialog_setup(benchmark):
    if benchmark == 'cvdn':
        return True
    elif benchmark == 'dialnav':
        return True
    else:
        return False


def _attach_observed_graph(navigator, obs, indices):
    """Copy only Navigator-observed graph nodes into local grounding inputs."""
    navigation_model = getattr(navigator, "navigation_model", navigator)
    visited_viewpoints = getattr(
        navigation_model, "visited_viewpoints", [])
    gmaps = getattr(navigation_model, "gmaps", [])
    local_obs = []
    for index in indices:
        item = dict(obs[index])
        visited = (
            list(visited_viewpoints[index])
            if index < len(visited_viewpoints)
            else [item["viewpoint"]]
        )
        gmap = gmaps[index] if index < len(gmaps) else None
        allowed = set(visited)
        allowed.add(item["viewpoint"])
        allowed.update(
            candidate["viewpointId"]
            for candidate in item.get("candidate", [])
        )
        positions = {}
        if gmap is not None:
            positions = {
                viewpoint: position
                for viewpoint, position in gmap.node_positions.items()
                if viewpoint in allowed
            }
        edges = []
        if gmap is not None:
            for source, targets in gmap.graph._dis.items():
                for target, distance in targets.items():
                    if source in allowed and target in allowed:
                        edges.append((source, target, float(distance)))
        item["_observed_viewpoints"] = visited
        item["_observed_positions"] = positions
        item["_observed_edges"] = edges
        local_obs.append(item)
    return local_obs


def dialNav(navigator,
            guide,
            mode,
            max_action_len=100, update_answer_behind=False,
            local_answer_model=None, local_answer_weight=0.4):
    navigator.set_next_batch()
    obs = navigator.get_obs()
    batch_size = len(obs)
    goals = guide.get_goals([ob['instr_id'] for ob in obs])
    profile = None
    if os.environ.get("DIALNAV_PROFILE_TIMING", "0") == "1":
        profile = getattr(navigator, "_dialnav_profile", None)
        if profile is None:
            profile = {}
            navigator._dialnav_profile = profile


    traj = [{
            'scan': ob['scan'],
            'start_pano': ob['viewpoint'],
            # 'gt_path': ob['gt_path'],
            'end_panos': goals[idx],
            'target': ob['instruction'],
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'navigation_detail': []
        } for idx, ob in enumerate(obs)]
    navigator.initialize_nav(obs)

    ask = np.array([False] * batch_size)


    question_seen_path = None
    answer_seen_path = None
    for step in range(max_action_len):
        nav_start = time.perf_counter() if profile is not None else None
        next_vp_ids, ended, nav_probs, instrucion_for_this_nav, nav_outs = navigator.get_next_action(step, obs)
        if profile is not None:
            _profile_add(profile, "nav_model_s", time.perf_counter() - nav_start)
        next_vp_ids_before_dialog = copy.deepcopy(next_vp_ids)
        ended_before_this_step = copy.deepcopy(ended)

        ## details
        nav_probs_cache = nav_probs.clone()
        c = torch.distributions.Categorical(nav_probs)
        c_cache = torch.distributions.Categorical(nav_probs_cache)

        if mode == 'navonly':
            ask = np.array([False] * batch_size)
        else:
            ask = navigator.wta(step, nav_probs, nav_outs)
        to_ask_indices = [index for index, value in enumerate(ask) if value and not ended[index]]
        need_dialog = len(to_ask_indices) > 0
        if need_dialog:
            print(f"[Step {step}] to_ask_indices: {to_ask_indices}")
        scanIds = [obs[i]['scan'] for i in range(batch_size)]
        viewpoints = [obs[i]['viewpoint'] for i in range(batch_size)]
        # goals = [obs[i]['gt_path'][-1] for i in range(batch_size)]

        if need_dialog:
            if mode == 'gt_loc':
                questions = ["Where should I go?" for i in range(batch_size)]
                localized_viewpoints = [obs[i]['viewpoint'] for i in range(batch_size)]
            else:
                question_start = time.perf_counter() if profile is not None else None
                questions, question_seen_path = navigator.ask(scanIds, viewpoints)
                if profile is not None:
                    _profile_add(
                        profile,
                        "question_generation_s",
                        time.perf_counter() - question_start,
                    )
                # print("ask1 questions", questions, question_seen_path)
                # questions, question_seen_path = navigator.ask2(scanIds, viewpoints, goals=goals)
                # print("ask 2 questions", questions, question_seen_path)
                localize_start = (
                    time.perf_counter() if profile is not None else None
                )
                if hasattr(guide, "localize_consistent"):
                    localized_viewpoints = guide.localize_consistent(
                        scanIds,
                        questions,
                        instr_ids=[obs[i]['instr_id'] for i in range(batch_size)],
                        active_indices=to_ask_indices,
                    )
                else:
                    localized_viewpoints = guide.localize(scanIds, questions)
                if profile is not None:
                    _profile_add(
                        profile,
                        "guide_localize_s",
                        time.perf_counter() - localize_start,
                    )
                # print("localized viewpoints", viewpoints)
            
            if hasattr(guide, "plan_answer"):
                answer_start = time.perf_counter() if profile is not None else None
                answers, answer_seen_path = guide.plan_answer(
                    scanIds,
                    localized_viewpoints,
                    questions,
                    goals,
                    instr_ids=[obs[i]['instr_id'] for i in range(batch_size)],
                    active_indices=to_ask_indices,
                    arrival_viewpoints=viewpoints,
                )
            else:
                answer_start = time.perf_counter() if profile is not None else None
                paths = [guide._choose_path(scanId, viewpoint, goal) for scanId, viewpoint, goal in zip(scanIds, localized_viewpoints, goals)]
                answers, answer_seen_path = guide.answer(scanIds, localized_viewpoints, paths)
            if profile is not None:
                _profile_add(
                    profile,
                    "guide_answer_s",
                    time.perf_counter() - answer_start,
                )
                _profile_add(profile, "dialog_requests", len(to_ask_indices))
            arrival_candidates = getattr(
                guide, "last_arrival_candidates", None
            )
            navigator.update_instruction(to_ask_indices, questions, answers, append_behind=update_answer_behind)

            # raise Exception("stop here")
            
            # update navigation actions with new dialog
            nav_start = time.perf_counter() if profile is not None else None
            next_vp_ids, ended, nav_probs, instrucion_for_this_nav, nav_outs = navigator.get_next_action(step, obs)
            if profile is not None:
                _profile_add(profile, "nav_model_s", time.perf_counter() - nav_start)
            if (
                os.environ.get("LOCAL_ANS_NAV") == "1"
                and local_answer_model is not None
                and to_ask_indices
            ):
                subset_obs = _attach_observed_graph(
                    navigator, obs, to_ask_indices)
                subset_answers = [answers[i] for i in to_ask_indices]
                try:
                    local_start = (
                        time.perf_counter() if profile is not None else None
                    )
                    local_answer_model.localize_local(
                        subset_obs, subset_answers)
                    if profile is not None:
                        _profile_add(
                            profile,
                            "local_grounding_s",
                            time.perf_counter() - local_start,
                        )
                    local_agent = getattr(local_answer_model, "agent", None)
                    local_logits = getattr(local_agent, "last_loc_logits", None)
                    local_nodes = getattr(local_agent, "last_loc_nodes", None)
                    if local_logits is not None and local_nodes is not None:
                        next_vp_ids, ended, nav_probs = (
                            navigator.apply_local_grounding(
                                to_ask_indices, local_logits, local_nodes,
                                obs, weight=local_answer_weight))
                except Exception as exc:
                    if profile is not None and local_start is not None:
                        _profile_add(
                            profile,
                            "local_grounding_s",
                            time.perf_counter() - local_start,
                        )
                    print(f"[LocalGrounding] skipped: {exc}")

        navigate_start = time.perf_counter() if profile is not None else None
        obs, paths = navigator.navigate(next_vp_ids, obs, ended, traj)
        if profile is not None:
            _profile_add(
                profile,
                "simulator_navigate_s",
                time.perf_counter() - navigate_start,
            )
            _profile_add(profile, "navigation_steps", 1)
        just_ended = ended & ~ended_before_this_step

        ### update trajectory log
        c = torch.distributions.Categorical(nav_probs)
        for i in range(batch_size):
            if ended[i] and not just_ended[i]:
                continue
            
            navigation_detail_item = {
                'nav_idx': step,
                'ask': False,
                'instruction': instrucion_for_this_nav[i],
                'gt_viewpoint': viewpoints[i],
                'next_vp_ids': next_vp_ids[i],
                'ended': ended[i],
                # 'nav_probs': nav_probs[i],
                'entropy': c.entropy()[i].item(),
            }
            if i in to_ask_indices:
                navigation_detail_item['ask'] = True
                navigation_detail_item['question'] = questions[i]
                navigation_detail_item['localized_viewpoint'] = localized_viewpoints[i]
                navigation_detail_item['answer'] = answers[i]
                if (
                    arrival_candidates is not None
                    and i < len(arrival_candidates)
                ):
                    navigation_detail_item['arrival_candidates'] = (
                        arrival_candidates[i]
                    )
                # navigation_detail_item['gt_viewpoint'] = viewpoints[i]
                if question_seen_path:
                    navigation_detail_item['question_seen_path'] = question_seen_path[i]
                if answer_seen_path:
                    navigation_detail_item['answer_seen_path'] = answer_seen_path[i]
                navigation_detail_item['vp_before_dialog'] = next_vp_ids_before_dialog[i]
                navigation_detail_item['entropy_before_dialog'] = c_cache.entropy()[i].item()
                navigation_detail_item['entropy_diff'] = navigation_detail_item['entropy'] - navigation_detail_item['entropy_before_dialog']
            traj[i]['navigation_detail'].append(navigation_detail_item)

            ## already processed in make_equiv_action
            # if not just_ended[i]:
            #     traj[i]['path'].append(paths[i])

        if all(ended):
            break

    visited_viewpoints = getattr(
        getattr(navigator, "navigation_model", navigator),
        "visited_viewpoints",
        [],
    )
    for index, item in enumerate(traj):
        if index < len(visited_viewpoints):
            item["visited_viewpoints"] = list(visited_viewpoints[index])
    return traj

def run(navigator, guide, max_action_len, mode, env_name, output_file, benchmark,
        local_answer_model=None, local_answer_weight=0.4):
    print("evaluating on env: ", env_name)
    start_time = time.time()

    # Set the target environment
    navigator.set_target_env(env_name)

    # Reset the data index to beginning of epoch. 
    navigator.reset_epoch()
    results = {}

    index = 1
    finished = False
    while not finished:
        print(f"Processing data {index}")
        index += 1
        # if index > 2:
        #     finished = True
        
        trajectories = dialNav(
            navigator, 
            guide, 
            max_action_len=max_action_len, 
            mode=mode,
            update_answer_behind=dialog_setup(benchmark),
            local_answer_model=local_answer_model,
            local_answer_weight=local_answer_weight,
        )
        for traj in trajectories:
            if traj['instr_id'] in results:
                finished = True
            if not finished:
                results[traj['instr_id']] = traj
            
        ### make output in list
        output = [{'instr_id': k, **v} for k, v in results.items()]
        ## save output to json
        with open(output_file, "w") as f:
            json.dump(output, f, default=lambda x: x.item() if isinstance(x, (bool, np.bool_)) else x)
    print("finished all trajectories", len(results))
    print("time taken: ", time.time() - start_time, "seconds")
    if os.environ.get("DIALNAV_PROFILE_TIMING", "0") == "1":
        profile = dict(getattr(navigator, "_dialnav_profile", {}))
        models = [
            ("guide_gtl", getattr(guide, "localization_model", None)),
            ("local_answer", local_answer_model),
        ]
        models.extend(
            (f"guide_plan_{index}", model)
            for index, model in enumerate(
                getattr(guide, "answer_grounding_models", [])
            )
        )
        for name, model in models:
            agent = getattr(model, "agent", None)
            values = getattr(agent, "profile_timing", None)
            if values is not None:
                for key, value in values.items():
                    profile[f"{name}_{key}"] = float(value)
        print("[Timing] " + json.dumps(profile, sort_keys=True))

    
    return output


def flatten_path(path):
    flat_path = []
    for step in path:
        if isinstance(step, list):
            flat_path.extend(flatten_path(step))
        else:
            flat_path.append(step)
    return flat_path


def add_target_coverage(output, evaluator, threshold):
    covered = 0
    for item in output:
        scan = item["scan"]
        visited = item.get("visited_viewpoints", [])
        targets = item.get("end_panos", [])
        item["target_visited"] = any(
            evaluator.shortest_distances[scan][viewpoint][target] <= threshold
            for viewpoint in visited
            for target in targets
            if viewpoint in evaluator.shortest_distances[scan]
            and target in evaluator.shortest_distances[scan][viewpoint]
        )
        covered += int(item["target_visited"])
    return covered / len(output) if output else 0.0


def make_submit_output(output):
    submit_output = []
    for item in output:
        submit_item = {k: v for k, v in item.items() if k != 'navigation_detail'}
        submit_item['path'] = flatten_path(item.get('path', []))
        ## remove start_pano, end_panos, nav_error, and gt_path from submit output
        submit_item.pop('start_pano', None)
        submit_item.pop('end_panos', None)
        submit_item.pop('nav_error', None)
        submit_item.pop('gt_path', None)
        submit_item.pop('visited_viewpoints', None)
        submit_item.pop('target_visited', None)

        dialog = []
        for detail in item.get('navigation_detail', []):
            if detail.get('ask'):
                dialog.append({
                    'nav_idx': detail.get('nav_idx'),
                    'question': detail.get('question'),
                    'answer': detail.get('answer'),
                })
        submit_item['dialog'] = dialog
        submit_output.append(submit_item)

    return submit_output
            
def setWta(wta_mode, navigation_model=None):
    if wta_mode.startswith('every'):
        print("Setting wta to every interval", wta_mode.split('_')[1])
        return FixedIntervalWtaModule(interval=int(wta_mode.split('_')[1]))
    elif wta_mode.startswith('hybrid'):
        parts = wta_mode.split('_')
        interval = int(parts[1]) if len(parts) > 1 else 4
        threshold = float(parts[2]) if len(parts) > 2 else 0.6
        print(f"Setting wta to hybrid interval={interval} threshold={threshold}")
        return HybridWtaModule(interval=interval, threshold=threshold)
    elif wta_mode.startswith('ct'):
        parts = wta_mode.split('_')
        threshold = float(parts[1])
        cap = int(parts[3]) if len(parts) > 3 and parts[2] == 'cap' else None
        min_step = int(parts[5]) if len(parts) > 5 and parts[4] == 'min' else 0
        if cap is not None:
            print(f"Setting wta to confidence thresholding with threshold {threshold}, cap {cap}, min_step {min_step}")
            return CappedConfidenceWtaModule(threshold=threshold, cap=cap, min_step=min_step)
        print("Setting wta to confidence thresholding with threshold", threshold)
        return ConfidenceThresholdingWtaModule(threshold=threshold)
    elif wta_mode == 'navigation_model':
        print("Setting wta to navigation model's internal wta")
        return navigation_model


def _configure_guide_plan(args, scans, guide_agent):
    """Load optional Guide-only answer planning models.

    The extra models are deliberately loaded only for GUIDE_PLAN_ENABLED
    experiments.  The Navigator receives only the answer text generated by
    ``guide_agent`` and never receives the selected path or candidate node.
    """
    if os.environ.get("GUIDE_PLAN_ENABLED", "0") != "1":
        return

    default_ckpts = ",".join([
        "/mnt/data-1/users/wangxiaofeng/dialnav_work/gtl_ans_ckpt_qa/final.pth",
        "/mnt/data-1/users/wangxiaofeng/dialnav_work/gtl_ans_ckpt2/final.pth",
    ])
    answer_ckpts = [
        path.strip()
        for path in os.environ.get(
            "GUIDE_PLAN_ANS_CKPTS", default_ckpts).split(",")
        if path.strip()
    ]
    for checkpoint in answer_ckpts:
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                f"Guide answer-grounding checkpoint not found: {checkpoint}")
        print(f"[GuidePlan] loading answer-grounding model {checkpoint}")
        guide_agent.answer_grounding_models.append(
            GraphVlnAgentModel(args.basepath, {
                "resume_file": checkpoint,
                "scan_list": scans,
            })
        )

    if os.environ.get("GUIDE_PLAN_USE_RERANK", "1") != "1":
        print("[GuidePlan] grounding-only mode; reranker is disabled")
        return

    rerank_checkpoint = os.environ.get(
        "GUIDE_PLAN_RERANK_CKPT",
        "/mnt/data-1/users/wangxiaofeng/dialnav_work/"
        "gtl_rerank_ckpt_hard/best.pth",
    )
    if not os.path.isfile(rerank_checkpoint):
        raise FileNotFoundError(
            f"Guide path-reranker checkpoint not found: {rerank_checkpoint}")
    print(f"[GuidePlan] loading path-reranker model {rerank_checkpoint}")
    rerank_model = LANA(args.basepath, {
        "scan_list": scans,
        # LANA's historical constructor supplies a default checkpoint when
        # ``resume_file`` is None.  Load the already available answer
        # checkpoint first, then replace its weights with the raw reranker
        # state dict below.
        "resume_file": args.ag_resume_file,
        "connectivity_dir": args.connectivity_dir,
        "bpe_path": args.qa_clip_tokenizer_path,
        "max_action_len": args.ag_max_answer_seen_path,
    }, type="ag")
    guide_agent.path_rerank_model = load_path_rerank_weights(
        rerank_model, rerank_checkpoint)
    print(
        f"[GuidePlan] enabled with {len(guide_agent.answer_grounding_models)} "
        "answer-grounding model(s)"
    )
    
def setAgents(args, target_envs, env_instructions, evaluator, scans):

    if args.nav_model == 'ScaleVLN':
        current_dir = os.path.dirname(os.path.abspath(__file__))
        modules_path = os.path.join(current_dir, '../../../modules/nav/ScaleVLN/map_nav_src')
        sys.path.insert(0, modules_path)
        ### Initialize Modules
        navigation_model_args = {
            'batch_size': args.batch_size, 
            'basepath': args.basepath, 
            'resume_file': args.nav_resume_file,
            'act_visited_nodes': args.nav_act_visited_nodes,
            'connectivity_dir': args.connectivity_dir,
            'wta_question_threshold': args.nav_wta_question_threshold,
        }
        navigation_model = ScaleVLNModel(args.basepath, navigation_model_args)
        navigation_model.eval()
        navigation_model.set_envs(target_envs, env_instructions)
    elif args.nav_model == 'DST':
        current_dir = os.path.dirname(os.path.abspath(__file__))
        modules_path = os.path.join(current_dir, '../../../modules/nav/DST/map_nav_src')
        sys.path.insert(0, modules_path)
        navigation_model_args = {
            'batch_size': args.batch_size, 
            'basepath': args.basepath, 
            'resume_file': args.nav_resume_file,
            'act_visited_nodes': args.nav_act_visited_nodes,
            'question_weight': args.nav_wta_question_threshold,
            'max_action_len': args.max_action_len,
        }
        navigation_model = DST(args.basepath, navigation_model_args)
        navigation_model.eval()
        navigation_model.set_envs(target_envs, env_instructions)
    else:
        raise ValueError(f"Invalid navigation model: {args.nav_model}")
    localization_model = None
    question_model = None
    answer_model = None
    wta_model = None
    if args.mode != 'navonly':
        wta_model = setWta(args.wta_mode, navigation_model)
        # wta_model = FixedIntervalWtaModule(interval=32) 
        # wta_model = ConfidenceThresholdingWtaModule(threshold=0.5) 
        # question_model = FixedQuestionGeneration(question='')
        # answer_model = FixedAnswerGeneration(response='Go straight')
        answer_model = LANA(args.basepath, {
            'scan_list': scans,
            'resume_file': args.ag_resume_file,
            'connectivity_dir': args.connectivity_dir,
            'bpe_path': args.qa_clip_tokenizer_path,
            'max_action_len': args.ag_max_answer_seen_path,
        }, type='ag')

        if args.mode != 'gt_loc':
            question_model = LANA(args.basepath, {
                'scan_list': scans,
                'resume_file': args.qg_resume_file,
                'connectivity_dir': args.connectivity_dir,
                'bpe_path': args.qa_clip_tokenizer_path,
            }, type='qg')

            if args.loc_model == 'GCN':
                localization_model = GCNLocModel(args.basepath, {
                    'eval_ckpt': args.loc_resume_file,
                    'panofeat_dir': args.loc_node_feats_dir,
                    'geodistance_file': args.loc_geodistance_nodes_path,
                    'connect_dir': args.connectivity_dir+"/",
                    'embedding_dir': args.loc_embedding_dir,
                    'bert_enc': args.loc_bert_enc,
                })
            elif args.loc_model == 'GTL':
                localization_model = GraphVlnAgentModel(args.basepath, {
                    'resume_file': args.loc_resume_file,
                    'scan_list': scans,
                })
            else:
                raise ValueError(f"Invalid localization model: {args.loc_model}")

    goals_by_instr = {
        item['instr_id']: item['end_panos']
        for env_name in target_envs
        for item in env_instructions[env_name]
    }
    targets_by_instr = {
        item['instr_id']: item['target']
        for env_name in target_envs
        for item in env_instructions[env_name]
    }
    env_infos = {
        "shortest_distances": evaluator.shortest_distances,
        "shortest_paths": evaluator.shortest_paths,
        "goals_by_instr": goals_by_instr,
        "targets_by_instr": targets_by_instr,
    }
    guide_agent = CompliantGuide(args, answer_model, localization_model, env_infos)
    _configure_guide_plan(args, scans, guide_agent)
    navigator_agent = ModularNavigator(args, navigation_model, wta_model, question_model)
    local_answer_model = None
    if os.environ.get("LOCAL_ANS_NAV") == "1":
        local_ckpt = os.environ.get("LOCAL_ANS_CKPT", "")
        if local_ckpt:
            local_answer_model = GraphVlnAgentModel(args.basepath, {
                'resume_file': local_ckpt,
                'scan_list': scans,
            })
        else:
            local_answer_model = localization_model
    return navigator_agent, guide_agent, local_answer_model



def main():
    _configure_cpu_threads()
    print("run parser")
    args = parse_args()
    target_envs = args.env_names.split(",")


    ### make output path
    os.makedirs(args.output_path, exist_ok=True)

    print("MAIN ARGS")
    print("args", args)
    args_log_file = f"{args.output_path}/args.txt"
    with open(args_log_file, "w") as f:
        f.write("--- MAIN ARGS --- \n")
        f.write("Time: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        for key, value in vars(args).items():
            if isinstance(value, (np.int64, np.float64)):
                value = value.item()
            f.write(f"{key}: {value}\n")
        f.write("\n\n")

    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0


    set_random_seed(args.seed + rank)

    tokenizer = get_tokenizer()
    env_instructions = load_instruction_data_universal(args, target_envs, tokenizer)
    if args.debug and 'val_seen' in env_instructions:
        env_instructions['val_seen'] = env_instructions['val_seen'][:16]
        # print(f"Debug mode enabled: using first 16 samples from val_seen only.")
        # target_envs = ['val_seen']
        env_instructions['val_unseen'] = env_instructions['val_unseen'][:16]
        env_instructions['test'] = env_instructions['test'][:16]

    ## set up evaluator
    scans = list(set([item['scan'] for env_name in target_envs for item in env_instructions[env_name]]))
    evaluator = Evaluator(
        args.connectivity_dir,
        scans,
        success_margin=args.success_margin,
        error_margin=args.error_margin,
    )

    navigator_agent, guide_agent, local_answer_model = setAgents(
        args, target_envs, env_instructions, evaluator, scans)

    with open(args_log_file, "a") as f:
        targets = {
            "navigation": navigator_agent.navigation_model.args,
        }
        if args.mode != 'navonly':
            targets['answer_generation'] = guide_agent.answer_model.args
        if args.mode == 'holistic':
            targets['question_generation'] = navigator_agent.question_generation_model.args
            targets['localization'] = guide_agent.localization_model.args
        
        f.write("--- NAVIGATOR ARGS --- \n")
        for key, value in targets.items():
            f.write(f"{key}: \n")
            for key, value in vars(value).items():
                if isinstance(value, (np.int64, np.float64)):
                    value = value.item()
                f.write(f"{key}: {value}\n")
            f.write("\n\n")


    submit_output = {}
    for env_name in target_envs:
        metrics_acc = {}
        avg_metrics_acc = {}
        output = run(
            navigator_agent, 
            guide_agent, 
            max_action_len=args.max_action_len,
            mode=args.mode,
            env_name=env_name,
            output_file=f"{args.output_path}/{env_name}.json",
            benchmark=args.benchmark,
            local_answer_model=local_answer_model,
            local_answer_weight=float(
                os.environ.get("LOCAL_ANS_WEIGHT", "0.4")),
        )

        target_coverage = add_target_coverage(
            output,
            evaluator,
            args.error_margin,
        )

        for item in output:
            item['nav_error'] = float(evaluator.get_shortest(item['scan'], item['path'][-1][-1], item['end_panos']))
            for detail in item['navigation_detail']:
                if 'localized_viewpoint' in detail:
                    detail['loc_error'] = float(evaluator.get_shortest(item['scan'], detail['gt_viewpoint'], [detail['localized_viewpoint']]))


        ## save output to json
        with open(f"{args.output_path}/{env_name}.json", "w") as f:
            json.dump(output, f, default=lambda x: x.item() if isinstance(x, (bool, np.bool_)) else x)

        submit_output[env_name]=make_submit_output(output)

        avg_metrics, metrics = evaluator.eval_metrics(output)
        avg_metrics['target_coverage'] = target_coverage
        metrics_acc[env_name] = metrics
        avg_metrics_acc[env_name] = avg_metrics
        avg_metrics_acc[env_name]['Agg'] = f"{','.join([str(round(avg_metrics_acc[env_name][key], 2)) for key in ['sr','oracle_sr','spl','nav_error','steps','dtc','le']])}"

        with open(f"{args.output_path}/avg_metrics_{env_name}.json", "w") as f:
            json.dump({'avg_metrics_acc': avg_metrics_acc[env_name]}, f, default=lambda x: x.item() if isinstance(x, (np.int64, np.float64)) else x)
        with open(f"{args.output_path}/metrics_{env_name}.json", "w") as f:
            json.dump({'metrics_acc': metrics_acc[env_name]}, f, default=lambda x: x.item() if isinstance(x, (np.int64, np.float64)) else x)

    with open(f"{args.output_path}/submit.json", "w") as f:
        json.dump(submit_output, f, default=lambda x: x.item() if isinstance(x, (bool, np.bool_)) else x)


if __name__ == '__main__':
    main()
