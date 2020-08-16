from collections import OrderedDict
import numpy as np
import pprint

from starter_code.interfaces.tree import TransformNode, find_first_nonright_ancestor, find_leftmost_descendant, is_root
from starter_code.infrastructure.log import log_string

def is_hierarchical(episode_data):
    return not episode_data[0].hierarchy_info.leaf


def flatten_rewards(episode_data):
    reward_chain = []
    for step in episode_data:
        if step.hierarchy_info.transformation_type in ['literal', 'function']:
            reward_chain.append(step.hierarchy_info.get_reward())
        elif step.hierarchy_info.transformation_type == 'option':
            reward_chain.extend(flatten_rewards(step.hierarchy_info.get_path_data()))
        else:
            assert False
    return reward_chain 


def build_interval_tree(episode_data, root=None):
    if root is None:
        root = TransformNode(
            id_num=np.inf, 
            transformation_type='option', 
            path_data=episode_data)
        root.set_before(0)
        root.set_sibling_index(0)

    # walk through the tree
    i = root.before
    for j, step in enumerate(episode_data):
        child = step.hierarchy_info
        child.set_before(i)
        if step.hierarchy_info.transformation_type == 'literal':
            child.set_after(i+1)
        elif step.hierarchy_info.transformation_type == 'option':
            child = build_interval_tree(child.path_data, root=child)
        elif step.hierarchy_info.transformation_type == 'function':
            child.set_after(i+1)
        else:
            assert False
        i = child.after
        child.set_parent(root)
        child.set_sibling_index(j)  # this child is the jth child
        assert child is root.get_children_nodes()[j]
    root.set_after(i)
    return root


def set_transformation_ids(parent):
    """ This works for two levels of hierarchy """
    path_data = parent.get_path_data()
    for i, step in enumerate(path_data):
        child = step.hierarchy_info
        child.set_current_transformation_id(parent.id_num)  # good
        if i == len(path_data) - 1:
            first_nonright_ancestor = find_first_nonright_ancestor(child)
            assert not first_nonright_ancestor is child  # have to go up at least one level
            if is_root(first_nonright_ancestor):
                child.set_next_transformation_id(parent.id_num)
            else:
                ancestor_parent = first_nonright_ancestor.parent
                ancestor_siblings = ancestor_parent.get_children_nodes()
                next_ancestor_sibling = ancestor_siblings[first_nonright_ancestor.sibling_index+1]
                leftmost_descendant = find_leftmost_descendant(next_ancestor_sibling)
                assert leftmost_descendant.before == child.after
                child.set_next_transformation_id(leftmost_descendant.parent.id_num)
        else:
            child.set_next_transformation_id(parent.id_num)
        if not child.leaf:
            set_transformation_ids(child)


def get_subreturns_matrix(reward_chain, gamma):
    """
    The diagonal is the reward
    """
    T = len(reward_chain)
    subreturns_matrix = np.empty((T, T))
    subreturns_matrix[:] = np.NaN

    for i in range(T):
        prev_subreturn = 0
        for j in range(i, T):
            subreturn = prev_subreturn + gamma**(j-i)*reward_chain[j]
            subreturns_matrix[i,j] = subreturn
            prev_subreturn = subreturn
    return subreturns_matrix


def redistribute_rewards_recursive(episode_data, subreturns_matrix):
    """ This is non-recursive and can only handle a two-level hierarchy """
    for i, step in enumerate(episode_data):
        before = step.hierarchy_info.before
        after = step.hierarchy_info.after

        # set rewards
        step.set_reward(subreturns_matrix[before, after-1])

        # recurse
        if not step.hierarchy_info.leaf:
            redistribute_rewards_recursive(
                step.hierarchy_info.path_data, subreturns_matrix)

    return episode_data


#################################################################
# visualization
def visualize_episode_data(episode_data):
    for t, step_data in enumerate(episode_data):
        try:
            reward = step_data.hierarchy_info.get_reward()
        except:
            reward = 'not set yet'
        step_dict = OrderedDict(
            t='{}\t'.format(t),
            state=np.argmax(step_data.state),
            action=step_data.action,
            next_state=np.argmax(step_data.next_state),
            reward='{}\t'.format(reward),
            mask=step_data.mask)
        print(log_string(step_dict))
        print('--'*20)

        if not step_data.hierarchy_info.leaf:
            for tt, baby_step_data in enumerate(step_data.hierarchy_info.path_data):
                step_dict = OrderedDict(
                    tt='{}\t'.format(tt),
                    state=np.argmax(baby_step_data.state),
                    action=baby_step_data.action,
                    next_state=np.argmax(baby_step_data.next_state),
                    reward='{}\t'.format(baby_step_data.reward),
                    mask=baby_step_data.mask)
                print('\t'+log_string(step_dict))


def visualize_hrl_finish_episode(episode_data, interval_tree, reward_chain, subreturns_matrix):
    print('episode_data')
    for step in episode_data:
        print(step)
    print('interval_tree')
    pprint.pprint(interval_tree.get_interval_dict())
    print('rewards')
    print(reward_chain)
    print('length of episode data: {}'.format(len(episode_data)))
    print('subreturns_matrix')
    for row in subreturns_matrix:
        print('\t'.join('{:.3f}'.format(x) for x in list(row)))

#################################################################