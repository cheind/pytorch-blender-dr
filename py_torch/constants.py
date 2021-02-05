GROUPS = [  # assign objects to groups (classes)
    [1, 2, 3, 4],
    [5, 6, 7, 8, 9],
    [10, 11, 12,],
    [13, 14, 15, 16, 17, 18],
    [19, 20, 21, 22, 23, 24],
    [25, 26, 27, 28, 29, 30],
]
NAMES = [f"{i}" for i in range(len(GROUPS))]
CATEGORIES = [{"id": id, "name": name} for id, name in enumerate(NAMES)]
# to map the 1 to 30 original classes to new classes of 0 to 5
MAPPING = {old_cls_id: new_cls_id for new_cls_id, group in enumerate(GROUPS) 
                                        for old_cls_id in group}
                                        