import matplotlib.pyplot as plt
from algorithm.classify.decisiontree.tree import TreeNode

decision_node_style = dict(boxstyle="sawtooth", fc="0.8")
leaf_node_style = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

__all__ = ['create_plot']

attr_dict = {0: 'age', 1: 'spectacle', 2: 'astigmatic', 3: 'tearRate'}


def get_leaf_num(node: TreeNode):
    if len(node.children) == 0:
        return 1
    res = 0
    for item in node.children.items():
        res += get_leaf_num(item[1])
    return res
    # return len(node.children)


def get_tree_depth(node):
    if len(node.children) == 0:
        return 0
    max_depth = 0
    for item in node.children.items():
        max_depth = max(max_depth, get_tree_depth(item[1]))
    return max_depth + 1


def plot_node(node_name, center_pt, parent_pt, node_type):
    create_plot.ax1.annotate(node_name, xy=parent_pt, xycoords='axes fraction',
                             xytext=center_pt, textcoords='axes fraction',
                             va="center", ha="center", bbox=node_type, arrowprops=arrow_args)


def plot_path_text(cntr_pt, parent_pt, text):
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    create_plot.ax1.text(x_mid, y_mid, text, va="center", ha="center", rotation=30, fontsize=10)


def plot_tree(node, parent_pt, attr_value):
    num_leaf = get_leaf_num(node)
    depth = get_tree_depth(node)
    cut_attr = attr_dict[node.ai]
    cntr_pt = (plot_tree.x + (1.0 + float(num_leaf)) / 2.0 / plot_tree.w, plot_tree.y)
    plot_path_text(cntr_pt, parent_pt, attr_value)
    plot_node(cut_attr, cntr_pt, parent_pt, decision_node_style)
    plot_tree.y = plot_tree.y - 1.0 / plot_tree.d
    for item in node.children.items():
        if item[1].children:
            plot_tree(item[1], cntr_pt, item[0])  # recursion
        else:
            plot_tree.x = plot_tree.x + 1.0 / plot_tree.w
            plot_node(item[1].label, (plot_tree.x, plot_tree.y), cntr_pt, leaf_node_style)
            plot_path_text((plot_tree.x, plot_tree.y), cntr_pt, item[0])
    plot_tree.y = plot_tree.y + 1.0 / plot_tree.d

def create_plot(root):
    fig = plt.figure(1, figsize=(12, 6), facecolor='white')
    fig.clf()
    ax_props = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **ax_props)
    # createPlot.ax1 = plt.subplot(111, frameon=False)
    plot_tree.w = float(get_leaf_num(root))
    plot_tree.d = float(get_tree_depth(root))
    plot_tree.x = -0.5 / plot_tree.w
    plot_tree.y = 1.0
    plot_tree(root, (0.5, 1.0), '')
    plt.show()

