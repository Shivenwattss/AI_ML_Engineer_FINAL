            "model_id": model_id,
            "generated_website_url": f"/models/{model_id}",
            "generated_api_url": f"/api/models/{model_id}/predict",
            "generated_api_docs": "/docs"
        }

    except Exception as e:

        raise HTTPException(
            400,
            str(e)
        )


# ============================================================
# RAG MEMORY
# ============================================================

@app.get("/memory")
def memory():

    return {
        "experiments": load()
    }


# ============================================================
# AI CHAT USING OLLAMA
# ============================================================


@app.post("/chat")
async def chat(message: str = Form(...)):
    import os
    import joblib

    info = {}

    # Get information about the current trained model
    if Path("models/best_model.joblib").exists():
        b = joblib.load("models/best_model.joblib")
        info = {
            "task": b.get("task"),
            "target": b.get("target"),
            "features": b.get("features", [])
        }

    try:
        api_key = os.getenv("OLLAMA_API_KEY")

        if not api_key:
            return {
                "answer": "Ollama Cloud is not configured. Check OLLAMA_API_KEY in Render Environment.",
                "llm_connected": False
            }

        # Prepare the chat prompt
        p = f"""
You are the AI ML Engineer assistant.

Current trained model:
{json.dumps(info, default=str)}

Previous experiments:
{json.dumps(load()[-5:], default=str)}

User question:
{message}

Answer clearly and only use information available in the context.
"""

        # Send the request to Ollama Cloud
        r = requests.post(
            "https://ollama.com/api/chat",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": os.getenv("OLLAMA_CLOUD_MODEL", "gemma4:31b"),
                "messages": [
                    {"role": "user", "content": p}
                ],
                "stream": False
            },
            timeout=(10, 120)
        )

        r.raise_for_status()
        data = r.json()
        answer = data["message"]["content"]

        return {
            "answer": answer,
            "llm_connected": True,
            "llm": "Ollama Cloud"
        }

    except requests.RequestException as e:
        return {
            "answer": (
                "Could not connect to Ollama Cloud. "
                f"Error: {str(e)[:300]}"
            ),
            "llm_connected": False
        }

    except (ValueError, KeyError, TypeError) as e:
        return {
            "answer": f"Unexpected Ollama response: {str(e)[:300]}",
            "llm_connected": False
        }

    except Exception as e:
        return {
            "answer": f"Chat error: {type(e).__name__}: {str(e)[:300]}",
            "llm_connected": False
        }

