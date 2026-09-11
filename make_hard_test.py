#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成「困难测试集」：人工编写的口语化句子，刻意偏离训练模板
（倒装、省略、生僻说法、无槽位等），用于检验模型在真实/未见说法上的泛化能力。
因为 train/test 都是模板合成，常规测试会虚高，这批才有区分度。

输出: data/hard_test.jsonl（格式与 train/test 完全一致）
用法: python make_hard_test.py
"""
import json
import os

# 每条: (utterance, intent, slots)
SAMPLES = [
    ("帮我整一份宫保鸡丁，多加辣", "点餐/下单", {"菜品": "宫保鸡丁", "数量": "一份", "口味偏好": "多放辣椒"}),
    ("烤鸭来半份，别放香菜", "点餐/下单", {"菜品": "烤鸭", "数量": "半份", "忌口": "不要香菜"}),
    ("明儿晚上七点左右，仨人，给安排个包间", "预订座位", {"就餐时间": "明晚7点", "就餐人数": "三位", "桌型": "包间"}),
    ("下周五中午12点，5位，想要靠窗户的", "预订座位", {"就餐时间": "周五中午12点", "就餐人数": "五位", "桌型": "靠窗位"}),
    ("你们这儿鱼香肉丝怎么卖啊", "查询菜单", {"菜品": "鱼香肉丝"}),
    ("有没有清淡点的菜，最好别太油", "查询菜单", {"口味偏好": "清淡点"}),
    ("给我找个人来聊聊，机器人说不明白", "转人工", {}),
    ("还是转人工吧", "转人工", {}),
    ("在国贸店吃的酸菜鱼有股怪味，申请退钱", "投诉/退款", {"菜品": "酸菜鱼", "门店位置": "国贸店"}),
    ("你们上菜太磨叽了，这单我不付了", "投诉/退款", {}),
    ("我卡里好像有张满减券，这回能用不", "会员/优惠", {"支付/优惠": "满100减20券"}),
    ("老顾客了，有积分能抵点钱吗", "会员/优惠", {"支付/优惠": "会员积分"}),
    ("中关村那个店一般几点打烊", "营业信息查询", {"门店位置": "中关村店"}),
    ("徐家汇店好找吗，具体在哪儿", "营业信息查询", {"门店位置": "徐家汇店"}),
    ("前面还有几桌啊，我这俩人", "排队取号", {"就餐人数": "两位"}),
    ("帮忙看看三里屯店现在排到几号了", "排队取号", {"门店位置": "三里屯店"}),
    ("今天股市咋样了", "闲聊/其他", {}),
    ("你会不会讲段子", "闲聊/其他", {}),
    ("可乐要大杯的，冰的", "点餐/下单", {"菜品": "可乐"}),
    ("我想订个位子，但还不确定几点到", "预订座位", {}),
]


def main():
    with open("schema.json", encoding="utf-8") as f:
        system_prompt = json.load(f)["system_prompt"]
    os.makedirs("data", exist_ok=True)
    out = []
    for utt, intent, slots in SAMPLES:
        assistant = json.dumps({"intent": intent, "slots": slots},
                               ensure_ascii=False, separators=(",", ":"))
        out.append({"messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": utt},
            {"role": "assistant", "content": assistant},
        ]})
    with open("data/hard_test.jsonl", "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"已写出 data/hard_test.jsonl（{len(out)} 条）")


if __name__ == "__main__":
    main()