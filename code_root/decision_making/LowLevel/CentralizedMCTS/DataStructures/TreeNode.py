class TreeNode:

    def __init__(self,
                 state,
                 parent,
                 depth,
                 is_terminal,
                 possible_actions,
                 action_to_get_here,
                 score,
                 num_visits,
                 children,
                 reward_to_here,
                 is_fully_expanded,
                 actions_taken_tracker,
                 event_at_node,
                 future_events_queue,
                 action_sequence_to_here=[],
                 # best_score_seen_from_here=float('-inf')
                 ):

        # self.best_score_seen_from_here = best_score_seen_from_here
        self.future_events_queue = future_events_queue
        self.event_at_node = event_at_node
        self.action_sequence_to_here = action_sequence_to_here
        self.actions_taken_tracker = actions_taken_tracker
        self.is_fully_expanded = is_fully_expanded
        self.reward_to_here = reward_to_here
        self.children = children
        self.num_visits = num_visits
        self.possible_actions = possible_actions
        self.is_terminal = is_terminal
        self.depth = depth
        self.parent = parent
        self.action_to_get_here = action_to_get_here
        self.score = score
        self.state = state