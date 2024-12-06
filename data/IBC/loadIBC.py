import _pickle as cPickle
import pandas as pd

if __name__ == '__main__':
    [lib, con, neutral] = cPickle.load(open('ibcData.pkl', 'rb'))

    data = pd.DataFrame(columns=["sentence", "label"])
    index = 0

    # how to access phrase labels for a particular tree
    ex_tree = lib[0]

    # can choose to only use the full sentence, or traverse the tree
    full_tree = False

    ids = [lib, con, neutral]

    # see treeUtil.py for the tree class definition
    for id in ids:
        for tree in id:
            for node in tree:

                # remember, only certain nodes have labels (see paper for details)
                if hasattr(node, 'label'):
                    data.loc[index, "sentence"] = node.get_words()
                    label = node.label

                    # if label == 'Conservative':
                    #     result = -1
                    # elif label == 'Neutral':
                    #     result = 0
                    # else:
                    #     result = 1
                    data.loc[index, "label"] = label

                    index += 1

                    if not full_tree:
                        break
            
    data.to_csv("ibc.csv", index=False)
