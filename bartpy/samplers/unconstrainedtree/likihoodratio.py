from typing import List

import numpy as np

from bartpy.model import Model
from bartpy.mutation import TreeMutation, GrowMutation, PruneMutation, ChangeMutation
from bartpy.node import LeafNode, DecisionNode, TreeNode
from bartpy.samplers.treemutation import TreeMutationLikihoodRatio
from bartpy.sigma import Sigma
from bartpy.tree import Tree


def left_and_right_sums(combined_node, left_node, right_node):
    if combined_node.data.y_sum_cache_up_to_date and left_node.data.y_sum_cache_up_to_date:
        combined_y_sum = combined_node.data.summed_y()
        left_y_sum = left_node.data.summed_y()
        right_y_sum = right_node.data.summed_y()
    else:
        combined_y_sum = combined_node.data.summed_y()
        left_y_sum = left_node.data.summed_y()
        right_y_sum = combined_y_sum - left_y_sum
    return combined_y_sum, left_y_sum, right_y_sum


def log_grow_ratio(combined_node: LeafNode, left_node: LeafNode, right_node: LeafNode, sigma: Sigma, sigma_mu: float):
    var = np.power(sigma.current_value(), 2)
    var_mu = np.power(sigma_mu, 2)
    n = combined_node.data.n_obsv
    n_l = left_node.data.n_obsv
    n_r = right_node.data.n_obsv

    first_term = (var * (var + n * sigma_mu)) / ((var + n_l * var_mu) * (var + n_r * var_mu))
    first_term = np.log(np.sqrt(first_term))
    combined_y_sum, left_y_sum, right_y_sum = left_and_right_sums(combined_node, left_node, right_node)

    left_resp_contribution = np.square(left_y_sum) / (var + n_l * sigma_mu)
    right_resp_contribution = np.square(right_y_sum) / (var + n_r * sigma_mu)
    combined_resp_contribution = np.square(combined_y_sum) / (var + n * sigma_mu)

    resp_contribution = left_resp_contribution + right_resp_contribution - combined_resp_contribution

    return first_term + ((var_mu / (2 * var)) * resp_contribution)


def log_change_ratio(old_node: DecisionNode, old_left_node: LeafNode, old_right_node: LeafNode,
                     node: LeafNode, left_node: LeafNode, right_node: LeafNode,
                     sigma: Sigma, sigma_mu: float):
    var = np.power(sigma.current_value(), 2)
    var_mu = np.power(sigma_mu, 2)
    n_1 = old_left_node.data.n_obsv
    n_1_star = left_node.data.n_obsv
    n_2 = old_right_node.data.n_obsv
    n_2_star = old_right_node.data.n_obsv
    combined_y_sum, left_y_sum, right_y_sum = left_and_right_sums(old_node, old_left_node, old_right_node)
    combined_y_sum_star, left_y_sum_star, right_y_sum_star = left_and_right_sums(node, left_node, right_node)

    first_term = (var / var_mu + n_1)/( var / var_mu + n_1_star) * (var / var_mu + n_2)/(var / var_mu + n_2_star)
    first_term = np.log(np.sqrt(first_term))
    left_contribution = np.square(left_y_sum) / (n_1 + var / var_mu)
    right_contribution = np.square(right_y_sum) / (n_2 + var / var_mu)
    left_contribution_star = np.square(left_y_sum_star) / (n_1_star + var / var_mu)
    right_contribution_star = np.square(right_y_sum_star) / (n_2_star + var / var_mu)
    contribution = left_contribution_star + right_contribution_star - left_contribution - right_contribution

    return first_term + contribution/(2 * var)


class UniformTreeMutationLikihoodRatio(TreeMutationLikihoodRatio):

    def __init__(self,
                 prob_method: List[float]=None):
        if prob_method is None:
            prob_method = [0.28, 0.28, 0.44]
        self.prob_method = prob_method

    def log_transition_ratio(self, tree: Tree, mutation: TreeMutation):
        if mutation.kind == "prune":
            mutation: PruneMutation = mutation
            return self.log_prune_transition_ratio(tree, mutation)
        if mutation.kind == "grow":
            mutation: GrowMutation = mutation
            return self.log_grow_transition_ratio(tree, mutation)
        if mutation.kind == "change":
            return 0
        else:
            raise NotImplementedError("kind {} not supported".format(mutation.kind))

    def log_tree_ratio(self, model: Model, tree: Tree, mutation: TreeMutation):
        if mutation.kind == "grow":
            mutation: GrowMutation = mutation
            return self.log_tree_ratio_grow(model, tree, mutation)
        if mutation.kind == "prune":
            mutation: PruneMutation = mutation
            return self.log_tree_ratio_prune(model, tree, mutation)
        if mutation.kind == "change":
            return 0

    def log_likihood_ratio(self, model: Model, tree: Tree, proposal: TreeMutation):
        if proposal.kind == "grow":
            proposal: GrowMutation = proposal
            return self.log_likihood_ratio_grow(model, proposal)
        if proposal.kind == "prune":
            proposal: PruneMutation = proposal
            return self.log_likihood_ratio_prune(model, proposal)
        if proposal.kind == "change":
            proposal: ChangeMutation = proposal
            return self.log_likelihood_ratio_change(model, proposal)
        else:
            raise NotImplementedError("Only prune, grow and change mutations supported")

    @staticmethod
    def log_likihood_ratio_grow(model: Model, proposal: TreeMutation):
        return log_grow_ratio(proposal.existing_node, proposal.updated_node.left_child, proposal.updated_node.right_child, model.sigma, model.sigma_m)

    @staticmethod
    def log_likihood_ratio_prune(model: Model, proposal: TreeMutation):
        return - log_grow_ratio(proposal.updated_node, proposal.existing_node.left_child, proposal.existing_node.right_child, model.sigma, model.sigma_m)

    @staticmethod
    def log_likelihood_ratio_change(model: Model, proposal: TreeMutation):
        return log_change_ratio(proposal.existing_node, proposal.existing_node.left_child, proposal.existing_node.right_child,
                                proposal.updated_node, proposal.updated_node.left_child,
                                proposal.updated_node.right_child, model.sigma, model.sigma_m)

    def log_grow_transition_ratio(self, tree: Tree, mutation: GrowMutation):
        prob_prune_selected = - np.log(n_prunable_decision_nodes(tree) + 1)
        prob_grow_selected = log_probability_split_within_tree(tree, mutation)

        prob_selection_ratio = prob_prune_selected - prob_grow_selected
        prune_grow_ratio = np.log(self.prob_method[1] / self.prob_method[0])

        return prune_grow_ratio + prob_selection_ratio

    def log_prune_transition_ratio(self, tree: Tree, mutation: PruneMutation):
        prob_grow_node_selected = safe_negative_log(n_splittable_leaf_nodes(tree) - 1)
        prob_split = log_probability_split_within_node(GrowMutation(mutation.updated_node, mutation.existing_node))
        prob_grow_selected = prob_grow_node_selected + prob_split

        prob_prune_selected = safe_negative_log(n_prunable_decision_nodes(tree))

        prob_selection_ratio = prob_grow_selected - prob_prune_selected
        grow_prune_ratio = np.log(self.prob_method[0] / self.prob_method[1])

        return grow_prune_ratio + prob_selection_ratio

    @staticmethod
    def log_tree_ratio_grow(model: Model, tree: Tree, proposal: GrowMutation):
        prob_chosen_split = log_probability_split_within_tree(tree, proposal)
        if model.prior_name in ["poly_splits","exponential_splits"]:
            denominator = log_probability_node_not_split(model, proposal.existing_node)

            prob_left_not_split = log_probability_node_not_split(model, proposal.updated_node.left_child)
            prob_right_not_split = log_probability_node_not_split(model, proposal.updated_node.right_child)
            prob_updated_node_split = log_probability_node_split(model, proposal.updated_node)
            numerator = prob_left_not_split + prob_right_not_split + prob_updated_node_split + prob_chosen_split
        elif model.prior_name == "cond_unif":
            K = len(tree.leaf_nodes)
            numerator = np.log(model.lam/(4*K-2)) + prob_chosen_split
            denominator = 0
        elif model.prior_name == "exp_prior":
            numerator = -model.c + prob_chosen_split
            denominator = 0
        else:
            return
        return numerator - denominator

    @staticmethod
    def log_tree_ratio_prune(model: Model, tree: Tree, proposal: PruneMutation):
        prob_chosen_split = log_probability_split_within_node(
            GrowMutation(proposal.updated_node, proposal.existing_node))
        if model.prior_name in ["poly_splits", "exponential_splits"]:
            numerator = log_probability_node_not_split(model, proposal.updated_node)

            prob_left_not_split = log_probability_node_not_split(model, proposal.existing_node.left_child)
            prob_right_not_split = log_probability_node_not_split(model, proposal.existing_node.left_child)
            prob_updated_node_split = log_probability_node_split(model, proposal.existing_node)
            prob_chosen_split = log_probability_split_within_node(GrowMutation(proposal.updated_node, proposal.existing_node))
            denominator = prob_left_not_split + prob_right_not_split + prob_updated_node_split + prob_chosen_split
        elif model.prior_name == "cond_unif":
            K = len(tree.leaf_nodes)
            # log gives negative for tree with single node, so don't suggest prune if single node
            denominator = np.log(model.lam/(4*K-6)) + prob_chosen_split
            numerator = 0
        elif model.prior_name == "exp_prior":
            denominator = -model.c + prob_chosen_split
            numerator = 0
        else:
            return
        return numerator - denominator


def n_prunable_decision_nodes(tree: Tree) -> int:
    """
    The number of prunable decision nodes
    i.e. how many decision nodes have two leaf children
    """
    return len(tree.prunable_decision_nodes)


def n_splittable_leaf_nodes(tree: Tree) -> int:
    """
    The number of splittable leaf nodes
    i.e. how many leaf nodes have more than one distinct values in their covariate matrix
    """
    return len(tree.splittable_leaf_nodes)


def log_probability_split_within_tree(tree: Tree, mutation: GrowMutation) -> float:
    """
    The log probability of the particular grow mutation being selected conditional on growing a given tree
    i.e.
    log(P(mutation | node)P(node| tree)

    """
    prob_node_chosen_to_split_on = safe_negative_log(n_splittable_leaf_nodes(tree))
    prob_split_chosen = log_probability_split_within_node(mutation)
    return prob_node_chosen_to_split_on + prob_split_chosen


def log_probability_split_within_node(mutation: GrowMutation) -> float:
    """
    The log probability of the particular grow mutation being selected conditional on growing a given node

    i.e.
    log(P(splitting_value | splitting_variable, node, grow) * P(splitting_variable | node, grow))
    """

    prob_splitting_variable_selected = safe_negative_log(mutation.existing_node.data.n_splittable_variables)
    splitting_variable = mutation.updated_node.most_recent_split_condition().splitting_variable
    splitting_value = mutation.updated_node.most_recent_split_condition().splitting_value
    prob_value_selected_within_variable = safe_log(mutation.existing_node.data.proportion_of_value_in_variable(splitting_variable, splitting_value))
    return prob_splitting_variable_selected + prob_value_selected_within_variable


def log_probability_node_split(model: Model, node: TreeNode):
    if model.prior_name == "poly_splits":
        return np.log(model.alpha * np.power(1 + node.depth, -model.beta))
    elif model.prior_name == "exponential_splits":
        return -node.depth*np.log(model.Gamma)


def log_probability_node_not_split(model: Model, node: TreeNode):
    if model.prior_name == "poly_splits":
        return np.log(1-model.alpha * np.power(1 + node.depth, -model.beta))
    elif model.prior_name == "exponential_splits":
        return np.log(1-np.power(1/model.Gamma, node.depth))

def safe_log(x):
    if x > 0:
        return np.log(x)
    elif x <= 0:
        return -np.inf

def safe_negative_log(x):
    if x > 0:
        return -np.log(x)
    elif x <= 0:
        return -np.inf