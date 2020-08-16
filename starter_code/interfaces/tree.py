class Node():
    def __init__(self, id_num, parent=None):
        self.id_num = id_num
        self.parent = parent

    def set_parent(self, parent_node):
        self.parent = parent_node

    def set_sibling_index(self, idx):
        self.sibling_index = idx

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


class IntervalNode(Node):
    def __init__(self, id_num):
        Node.__init__(self, id_num)

    def set_before(self, before):
        self.before = before

    def set_after(self, after):
        self.after = after

    def get_interval_dict(self):
        interval_dict = dict(
            id_num=self.id_num,
            before=self.before,
            after=self.after,
            sibling_index=self.sibling_index,
            )
        if hasattr(self, 'current_transformation_id'):
            interval_dict['current_transformation_id'] = self.current_transformation_id
            interval_dict['next_transformation_id'] = self.next_transformation_id
        children = self.get_children_nodes()
        if children is not None:
            interval_dict['children'] = [c.get_interval_dict() for c in children]
        return interval_dict


class TransformNode(IntervalNode):
    def __init__(self, id_num, transformation_type, path_data):
        IntervalNode.__init__(self, id_num)
        self.transformation_type = transformation_type

        # for distribute_transitions_recursive
        self.path_data = path_data

    def get_path_data(self):
        return self.path_data

    def get_reward(self):
        raise NotImplementedError

    def get_length(self):
        raise NotImplementedError

    def get_organism(self):
        raise NotImplementedError

    def get_children_nodes(self):
        return [step_info.get_hierarchy_info() for step_info in self.path_data]

    def is_parent_of_leaf(self):
        raise NotImplementedError

    def set_current_transformation_id(self, id_num):
        self.current_transformation_id = id_num

    def set_next_transformation_id(self, id_num):
        self.next_transformation_id = id_num

    def __repr__(self):
        return '{}-{}'.format(self.__class__.__name__, self.id_num)


class LiteralTransformNode(TransformNode):
    def __init__(self, id_num, path_data):
        TransformNode.__init__(self, id_num, 'literal', path_data)
        # for distribute_transitions_recursive
        self.leaf = True

    def get_reward(self):
        return self.path_data['reward']

    def get_length(self):
        return 1

    def get_children_nodes(self):
        return None

    def is_parent_of_leaf(self):
        return False


class OptionTransformNode(TransformNode):
    def __init__(self, id_num, organism, path_data):
        TransformNode.__init__(self, id_num, 'option', path_data)
        # for distribute_transitions_recursive
        self.leaf = False
        self.organism = organism

    def get_length(self):
        return len(self.path_data)

    def get_organism(self):
        return self.organism

    def get_children_nodes(self):
        return [step_info.get_hierarchy_info() for step_info in self.path_data]

    def is_parent_of_leaf(self):
        first_child = self.get_children_nodes()[0]
        return first_child.leaf


class FunctionTransformNode(LiteralTransformNode):
    def __init__(self, id_num, path_data):
        TransformNode.__init__(self, id_num, 'function', path_data)
        self.leaf = True

    def set_reward(self, reward):
        self.path_data['reward'] = reward


def find_leftmost_descendant(node):
    """
    Given a node, find its leftmost descendant
    """
    if node.leaf:
        return node
    else:
        leftmost_child = node.get_children_nodes()[0]
        return find_leftmost_descendant(leftmost_child)


def find_first_nonright_ancestor(node):
    # root
    if is_root(node):
        return node
    elif not node is node.parent.get_children_nodes()[-1]:
        return node
    else:
        return find_first_nonright_ancestor(node.parent)


def is_root(node):
    return node.parent is None
