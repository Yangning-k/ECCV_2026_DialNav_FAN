from interface.Navigation import Navigation
from interface.WTA import WTA
from interface.QuestionGeneration import QuestionGeneration
from interface.NavigatorAgent import NavigatorAgent

class ModularNavigator(NavigatorAgent):
    def __init__(self, args, navigation_model: Navigation, wta_model: WTA, question_generation_model: QuestionGeneration):
        super().__init__(args)
        self.args = args
        self.navigation_model = navigation_model
        self.wta_model = wta_model
        self.question_generation_model = question_generation_model

    ##### Navigation Functions #####
    def set_target_env(self, env_name):
        self.navigation_model.set_target_env(env_name)
    
    def reset_epoch(self):
        self.navigation_model.reset_epoch()
    
    def get_obs(self):
        return self.navigation_model.get_obs()
    
    def set_next_batch(self):
        self.navigation_model.set_next_batch()
    
    def initialize_nav(self, obs):
        self.navigation_model.initialize_nav(obs)
    
    def get_next_action(self, nav_idx, obs):
        return self.navigation_model.get_next_action(nav_idx, obs)
    
    def navigate(self, next_vp_ids, obs, just_ended, traj):
        return self.navigation_model.navigate(next_vp_ids, obs, just_ended, traj)
    
    def update_instruction(self, to_ask_indices, questions, answers, append_behind=False):
        self.navigation_model.update_instruction(to_ask_indices, questions, answers, append_behind)

    def set_arrival_judge_targets(self, targets):
        if hasattr(self.navigation_model, "set_arrival_judge_targets"):
            self.navigation_model.set_arrival_judge_targets(targets)

    def set_target_descriptions(self, descriptions):
        if hasattr(self.navigation_model, "set_target_descriptions"):
            self.navigation_model.set_target_descriptions(descriptions)

    def get_confirm_candidate_sets(self):
        if hasattr(self.navigation_model, "get_confirm_candidate_sets"):
            return self.navigation_model.get_confirm_candidate_sets()
        return None

    def apply_local_grounding(self, *args, **kwargs):
        return self.navigation_model.apply_local_grounding(*args, **kwargs)

    ##### WTA Functions #####
    def wta(self, step, nav_probs, nav_outs):
        if self.wta_model is None:
            raise ValueError("wta_model is not set")
        suppressed = getattr(self.navigation_model, "dialog_suppressed", None)
        confirm = getattr(self.navigation_model, "confirm_ask_request", None)
        if suppressed is not None and hasattr(self.wta_model, "set_blocked"):
            self.wta_model.set_blocked(
                [
                    bool(flag)
                    and not (confirm is not None and bool(confirm[index]))
                    for index, flag in enumerate(suppressed)
                ]
            )
        ask = self.wta_model.wta(step, nav_probs, nav_outs)
        if suppressed is None and confirm is None:
            return ask
        return [
            bool(confirm[index])
            if confirm is not None and bool(confirm[index])
            else bool(value)
            and (suppressed is None or not bool(suppressed[index]))
            for index, value in enumerate(ask)
        ]
    
    def ask(self, *args, **kwargs):
        if self.question_generation_model is None:
            raise ValueError("question_generation_model is not set")
        
        questions = self.question_generation_model.ask(*args, **kwargs)
        return questions

# class ModularNavigatorWta(ModularNavigator):
#     def __init__(self, args, navigation_model: Navigation, wta_model: None, question_generation_model: QuestionGeneration):
#         super().__init__(args, navigation_model, wta_model, question_generation_model)

#     def wta(self):
#         return self.navigation_model.wta()