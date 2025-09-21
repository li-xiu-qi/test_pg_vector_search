from pathlib import Path
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer


def find_modelscope_cache_path(model_id: str) -> Path | None:
    """尝试返回 ModelScope 的本地缓存路径（若存在），否则返回 None。

    ModelScope 默认缓存路径通常位于用户主目录下的
    ~/.cache/modelscope/hub/models/<owner>/<model>
    Windows 示例: C:\\Users\\user_name\\.cache\\modelscope\\hub\\models\\BAAI\\bge-m3
    """
    home = Path.home()
    # 常见 cache 路径
    candidates = []
    # 按 owner/name 构建常见缓存路径
    parts = model_id.split("/")
    base = home / ".cache" / "modelscope" / "hub" / "models"
    if len(parts) == 2:
        owner, name = parts
        candidates.append(base / owner / name)
    else:
        # 非标准格式，直接把整个 model_id 作为目录名尝试
        candidates.append(base / model_id)

    # 有时模型路径可能保留斜杠或被转义为一个名字，尝试直接替换
    candidates.append(base / model_id.replace("/", "-"))
    candidates.append(base / model_id.replace("/", "_"))

    for p in candidates:
        # 过滤掉非 Path 项（理论上候选项都是 Path）
        if not isinstance(p, Path):
            continue
        if p.exists():
            return p
    return None


def main():
    model_id = "BAAI/bge-m3"
    local_path = find_modelscope_cache_path(model_id)
    if local_path:
        print(f"检测到本地 ModelScope 缓存，使用本地模型路径：{local_path}")
        model_source = str(local_path)
    else:
        print(f"未检测到本地缓存，回退使用在线模型 id：{model_id}")
        model_source = model_id

    model = SentenceTransformer(model_source)
    texts = ["这是一个测试句子。", "Hello world"]
    embeddings = model.encode(texts, normalize_embeddings=True)
    print(f"Embeddings shape: {embeddings.shape}")

    # 将向量写入 PostgreSQL (pgvector)
    # 连接参数与 docker-compose 保持一致
    conn = psycopg2.connect(
        host="127.0.0.1",
        port=5432,
        dbname="vectordb",
        user="postgres",
        password="postgres",
    )
    conn.autocommit = True
    with conn:
        with conn.cursor() as cur:
            # 创建扩展（幂等）与表（示例: 1024 维，取自 bge-m3 输出）
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS items (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding vector(1024) NOT NULL
                );
                """
            )

            # 批量插入文本与向量
            rows = [
                (texts[i], embeddings[i].tolist()) for i in range(len(texts))
            ]
            # psycopg2 传输 pgvector: 用 Python 数组文本格式，形如 '[1,2,3]'
            rows = [(t, f"[{','.join(str(x) for x in vec)}]") for t, vec in rows]
            execute_values(
                cur,
                "INSERT INTO items (text, embedding) VALUES %s",
                rows,
            )
            print(f"已写入 {len(rows)} 条向量记录到 items 表。")

            # 简单相似度查询（使用 <-> L2 距离 或 inner product / cosine 需对应归一化）
            # 注意：当 normalize_embeddings=True 时，按 L2 距离排序与按余弦相似度排序等价，
            # 但“距离值”本身并不等于余弦距离（L2 与 cosine 的数值不同，仅排序一致）。
            q = "这是一个查询句子。"
            q_emb = model.encode([q], normalize_embeddings=True)[0]
            q_arr = f"[{','.join(str(x) for x in q_emb)}]"
            cur.execute(
                """
                SELECT id, text, embedding <-> %s AS distance
                FROM items
                ORDER BY distance ASC
                LIMIT 3;
                """,
                (q_arr,),
            )
            results = cur.fetchall()
            print("Top-3 相似结果（L2 距离）：")
            for rid, t, dist in results:
                print(f"  id={rid}, dist={float(dist):.6f}, text={t}")

            # 余弦相似度检索示例：使用 <=> 计算 cosine distance（值越小越相似）
            # 若需要打印相似度（cosine similarity），可用 1 - cosine_distance
            cur.execute(
                """
                SELECT id, text, embedding <=> %s AS cos_distance
                FROM items
                ORDER BY cos_distance ASC
                LIMIT 3;
                """,
                (q_arr,),
            )
            cos_results = cur.fetchall()
            print("Top-3 余弦检索结果：")
            for rid, t, cos_dist in cos_results:
                cos_sim = 1.0 - float(cos_dist)
                print(f"  id={rid}, cos_sim={cos_sim:.6f}, text={t}")

    conn.close()


if __name__ == "__main__":
    main()
