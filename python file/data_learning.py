import re

def split_text_into_blocks(evidence_base, block_size=200, step=100):
    blocks = []

    for key in evidence_base.keys():
        text = evidence_base[key]

        # 1️⃣ 按句子分割（保留标点）
        sentences = re.split(r'(?<=[。！？])', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # 计算平均每句的字符数，进而计算步进多少句
        avg_sentence_length = len(text) / len(sentences) if sentences else 1
        step_sentences = max(1, int(step / avg_sentence_length))

        sentence_start = 0  # 滑动窗口起始位置

        while sentence_start < len(sentences):
            current_block = []
            current_length = 0
            i = sentence_start

            # 组合句子，直到达到 block_size 限制
            while i < len(sentences) and current_length + len(sentences[i]) <= block_size:
                current_block.append(sentences[i])
                current_length += len(sentences[i])
                i += 1

            if current_block:
                blocks.append("".join(current_block))
            else:
                # 如果连单句都不能加入，直接将该句加入，避免死循环
                blocks.append(sentences[sentence_start])
                i = sentence_start + 1

            # 滑动窗口移动
            sentence_start += step_sentences

    return blocks


evidence_base = {
    "doc1": "今天天气很好，我们去公园玩吧！小明和小红也想去。到了公园，他们玩得很开心。"
}

blocks = split_text_into_blocks(evidence_base, block_size=20, step=10)
for i, block in enumerate(blocks):
    print(f"Block {i+1}: {block}")
