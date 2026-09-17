import json
import sqlite3
from pathlib import Path

def check_file(path_str, description):
    path = Path(path_str)
    print(f"\n{'='*60}")
    print(f" {description}")
    print(f" Path: {path}")
    print(f"{'='*60}")
    
    if not path.exists():
        print("STATUS: File does not exist yet.")
        return False
    
    print(f"STATUS: Exists ({path.stat().st_size} bytes)")
    return True

def main():
    print("RESEARCH AGENT - MEMORY & STORAGE INSPECTOR")
    
    # 1. Main Research Database (Phase 1/2 outputs)
    db_path = Path("data/research_agent.db")
    if check_file(db_path, "MAIN RESEARCH DATABASE (Saved Runs)"):
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            print(f"Tables found: {[t[0] for t in tables]}")
            
            if ("runs",) in tables:
                count = cursor.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
                print(f"Total saved runs: {count}")
                if count > 0:
                    print("\nLast 3 runs:")
                    rows = cursor.execute("SELECT request_id, topic, status, total_sources, created_at FROM runs ORDER BY created_at DESC LIMIT 3").fetchall()
                    for r in rows:
                        print(f"  - [{r[2]}] {r[1][:40]} | Sources: {r[3]} | ID: {r[0][:8]}...")
            conn.close()
        except Exception as e:
            print(f"ERROR reading DB: {e}")

    # 2. Agent Long-Term Memory DB (Phase 3)
    agent_db = Path("data/agent_memory.db")
    if check_file(agent_db, "AGENT LONG-TERM MEMORY"):
        try:
            conn = sqlite3.connect(str(agent_db))
            cursor = conn.cursor()
            tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            print(f"Tables found: {[t[0] for t in tables]}")
            for t in tables:
                count = cursor.execute(f"SELECT COUNT(*) FROM {t[0]}").fetchone()[0]
                print(f"  - {t[0]}: {count} rows")
            conn.close()
        except Exception as e:
            print(f"ERROR reading DB: {e}")

    # 3. Agent Episodic Memory (Phase 3)
    ep_path = Path("logs/agent_episodes.jsonl")
    if check_file(ep_path, "AGENT EPISODIC MEMORY (Past Experiences)"):
        try:
            lines = ep_path.read_text(encoding="utf-8").strip().split("\n")
            print(f"Total episodes recorded: {len(lines)}")
            print("\nLast 3 episodes:")
            for line in lines[-3:]:
                data = json.loads(line)
                task = data.get("task", "unknown")[:40]
                success = "SUCCESS" if data.get("success") else "FAILED"
                score = data.get("quality_score", 0)
                tokens = data.get("tokens_used", 0)
                print(f"  - [{success}] {task} | Score: {score:.2f} | Tokens: {tokens}")
        except Exception as e:
            print(f"ERROR reading JSONL: {e}")

    # 4. Vector Store (RAG)
    vs_path = Path("data/vector_store.json")
    if check_file(vs_path, "VECTOR STORE (RAG Memory)"):
        try:
            data = json.loads(vs_path.read_text(encoding="utf-8"))
            docs = data.get("documents", [])
            print(f"Total documents indexed: {len(docs)}")
            print(f"Last updated: {data.get('updated_at', 'unknown')}")
            if docs:
                print("\nFirst 3 documents:")
                for doc in docs[:3]:
                    print(f"  - {doc.get('title', 'untitled')[:50]} ({doc.get('platform', '?')})")
        except Exception as e:
            print(f"ERROR reading JSON: {e}")

    print(f"\n{'='*60}")
    print(" INSPECTION COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()