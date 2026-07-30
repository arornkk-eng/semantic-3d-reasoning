"""中文→英文物体名翻译：字典映射 + 模糊匹配。

用于将用户的中文查询转为 Grounding DINO 可识别的英文词。
"""

import re

# 常见室内物体中英映射（按使用频率排序）
_CN_TO_EN: dict[str, str] = {
    # 餐具/厨房
    "水杯": "water cup",
    "杯子": "cup",
    "玻璃杯": "glass cup",
    "马克杯": "mug",
    "保温杯": "thermos cup",
    "水瓶": "water bottle",
    "瓶子": "bottle",
    "碗": "bowl",
    "盘子": "plate",
    "碟子": "plate",
    "筷子": "chopsticks",
    "勺子": "spoon",
    "叉子": "fork",
    "刀": "knife",
    "茶壶": "teapot",
    "水壶": "kettle",
    "锅": "pot",
    "平底锅": "pan",
    # 家具
    "椅子": "chair",
    "凳子": "stool",
    "桌子": "table",
    "书桌": "desk",
    "餐桌": "dining table",
    "茶几": "coffee table",
    "沙发": "sofa",
    "床": "bed",
    "书架": "bookshelf",
    "书柜": "bookcase",
    "柜子": "cabinet",
    "抽屉": "drawer",
    "衣柜": "wardrobe",
    "床头柜": "nightstand",
    # 电子设备
    "电脑": "laptop",
    "笔记本电脑": "laptop",
    "笔记本": "laptop",
    "手机": "phone",
    "平板": "tablet",
    "电视": "tv",
    "显示器": "monitor",
    "屏幕": "monitor",
    "键盘": "keyboard",
    "鼠标": "mouse",
    "音箱": "speaker",
    "耳机": "headphones",
    "遥控器": "remote control",
    # 装饰/杂物
    "台灯": "lamp",
    "灯": "lamp",
    "落地灯": "floor lamp",
    "吊灯": "ceiling lamp",
    "钟": "clock",
    "闹钟": "alarm clock",
    "花瓶": "vase",
    "花盆": "flower pot",
    "植物": "plant",
    "盆栽": "potted plant",
    "相框": "picture frame",
    "画": "painting",
    "镜子": "mirror",
    "地毯": "rug",
    "窗帘": "curtain",
    "抱枕": "pillow",
    "枕头": "pillow",
    "毛毯": "blanket",
    # 人物/动物
    "人": "person",
    "猫": "cat",
    "狗": "dog",
    # 食物
    "苹果": "apple",
    "香蕉": "banana",
    "橙子": "orange",
    # 箱包
    "书包": "backpack",
    "背包": "backpack",
    "手提包": "handbag",
    "箱子": "box",
    "行李箱": "suitcase",
    # 文具
    "书": "book",
    "笔": "pen",
    "铅笔": "pencil",
    "笔记本": "notebook",
    # 其他
    "门": "door",
    "窗户": "window",
    "垃圾桶": "trash can",
    "扫帚": "broom",
    "雨伞": "umbrella",
    "钥匙": "key",
    "眼镜": "glasses",
    "帽子": "hat",
    "鞋子": "shoe",
    "衣服": "clothing",
    "包": "bag",
    "玩具": "toy",
    "球": "ball",
    # 材质/特征（兜底）
    "不锈钢": "stainless steel",
    "金属": "metal",
    "塑料": "plastic",
    "玻璃": "glass",
    "木质": "wooden",
    "陶瓷": "ceramic",
    "白色": "white",
    "黑色": "black",
    "红色": "red",
    "蓝色": "blue",
    "绿色": "green",
}


def translate_query(text: str) -> list[str]:
    """将用户查询转为英文候选词列表。

    按优先级返回多个候选：精确匹配 → 部分匹配 → 原词。
    Grounding DINO 会尝试所有候选，取检测结果最好的。

    Args:
        text: 用户输入，如 "水杯" 或 "白色的杯子"

    Returns:
        英文候选词列表，如 ["water cup", "white cup", "cup"]
    """
    text = text.strip()
    if not text:
        return ["object"]

    # 1. 全英文 → 直接返回
    if all(ord(c) < 128 for c in text):
        return [text]

    # 2. 精确匹配
    if text in _CN_TO_EN:
        return [_CN_TO_EN[text]]

    # 3. 提取中文词并逐个翻译（处理 "白色的杯子" → white + cup）
    candidates: list[str] = []
    remaining = text

    # 按词长降序匹配（优先长词）
    sorted_keys = sorted(_CN_TO_EN.keys(), key=len, reverse=True)
    for cn_word in sorted_keys:
        if cn_word in remaining:
            candidates.append(_CN_TO_EN[cn_word])
            remaining = remaining.replace(cn_word, " ", 1)

    if candidates:
        # 组合：如果用户写 "白色的杯子" → ["white cup"]
        combined = " ".join(candidates)
        return [combined] + candidates

    # 4. 完全没匹配 → 拆单个字尝试 + 兜底
    single_words = []
    for ch in text:
        if ch in _CN_TO_EN:
            single_words.append(_CN_TO_EN[ch])
    if single_words:
        return [" ".join(single_words)] + single_words

    # 5. 最终兜底：原词（可能 Grounding DINO 恰好认识）
    return [text]
