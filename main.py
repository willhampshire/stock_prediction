from fastapi import FastAPI, HTTPException
import uvicorn
from typing import Optional, Any
from state import json_state


def get_nested_data(d: Any, path: str):
    keys = path.split('.')
    for key in keys:
        if '[' in key and ']' in key:
            key, index = key[:-1].split('[')
            d = d.get(key, [])
            index = int(index)
            if index >= len(d):
                raise HTTPException(status_code=404, detail="Index out of range")
            d = d[index]
        else:
            d = d.get(key)
            if d is None:
                raise HTTPException(status_code=404, detail="Path not found")
    return d

app = FastAPI()

@app.get("/")
async def get(query: Optional[str] = None):
    json = json_state.json
    print(f"get, json_state {json}")
    if query:
        try:
            result = get_nested_data(json, query)
            return result
        except HTTPException as e:
            return {"error": e.detail}
    return json




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


