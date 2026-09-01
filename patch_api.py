import sys

def main():
    filename = 'api_server.py'
    with open(filename, 'r', newline='') as f:
        lines = f.readlines()
    
    # 1. Replace startup_event (lines 197-201 zero-indexed? Let's compute)
    # We'll find the line index for '@app.on_event("startup")'
    startup_idx = None
    for i, line in enumerate(lines):
        if line.strip() == '@app.on_event("startup")':
            startup_idx = i
            break
    if startup_idx is None:
        print("Could not find startup_event")
        sys.exit(1)
    # The function consists of the decorator line, the def line, docstring, and two print lines.
    # We'll replace from startup_idx to the line before the next '@app.on_event' (shutdown) or end.
    # Let's find the end index: look for the next line that starts with '@app.on_event' after startup_idx.
    end_idx = None
    for i in range(startup_idx+1, len(lines)):
        if lines[i].strip().startswith('@app.on_event'):
            end_idx = i
            break
    if end_idx is None:
        end_idx = len(lines)
    # We want to replace lines[startup_idx:end_idx] with new block.
    # Build new block
    indent = '    '  # assuming 4 spaces
    new_startup = [
        lines[startup_idx],  # '@app.on_event("startup")\\n'
        lines[startup_idx+1], # 'async def startup_event():\\n'
        lines[startup_idx+2], # '    """Initialize services on startup."""\\n'
        indent + 'print("RepoScore API v2.0 starting up...")\\n',
        indent + 'try:\\n',
        indent + '    print(f"Cache: {get_cache().redis.connection_pool.connection_kwargs}")\\n',
        indent + 'except Exception as e:\\n',
        indent + '    print(f"Cache: Redis unavailable ({e})")\\n'
    ]
    # Note: we need to ensure we don't add extra blank lines; the original had a blank line after the second print? Actually after the second print there was a blank line before the shutdown decorator.
    # We'll keep the blank line by not adding it; the next line after the block will be whatever was at end_idx (which is the shutdown decorator line).
    # So we need to ensure we don't add an extra newline at the end of the block; the last line already ends with newline.
    # Replace
    lines = lines[:startup_idx] + new_startup + lines[end_idx:]
    
    # 2. Replace submit_analysis function
    # Find '@app.post("/api/jobs")'
    post_idx = None
    for i, line in enumerate(lines):
        if line.strip() == '@app.post("/api/jobs")':
            post_idx = i
            break
    if post_idx is None:
        print("Could not find submit_analysis")
        sys.exit(1)
    # Find the end of the function: look for the next line that starts with '@app' or 'async def' (the next function) after the post_idx.
    # Actually the function ends before the line 'async def analyze_repo'
    # We'll find the line index of 'async def analyze_repo'
    analyze_idx = None
    for i in range(post_idx+1, len(lines)):
        if lines[i].strip().startswith('async def analyze_repo'):
            analyze_idx = i
            break
    if analyze_idx is None:
        print("Could not find analyze_repo function")
        sys.exit(1)
    # We'll replace lines[post_idx:analyze_idx] with new submit_analysis.
    # Build new block.
    # We need to keep the decorator line and the def line and docstring.
    # We'll keep lines[post_idx] through lines[post_idx+2] (decorator, def, docstring) and then replace the rest.
    # Let's get the indent of the function body: look at line post_idx+3 (the first line of the function body).
    body_indent = len(lines[post_idx+3]) - len(lines[post_idx+3].lstrip())
    indent_str = lines[post_idx+3][:body_indent]
    
    new_submit = [
        lines[post_idx],  # '@app.post("/api/jobs")\\n'
        lines[post_idx+1], # 'async def submit_analysis(owner: str = Query(...), repo: str = Query(...), \\n'
        lines[post_idx+2], #                          '                          threshold: float = Query(0.3, ge=0.0, le=1.0)):\\n'
        lines[post_idx+3], # '    """\\n'
        lines[post_idx+4], #     \"\"\"Submit a repository for quality analysis.\\n'
        lines[post_idx+5], #     \"\"\"Returns a job ID for tracking progress.\\n'
        lines[post_idx+6], #     \"\"\"\\n'
        indent_str + '    job_id = str(uuid.uuid4())\\n',
        indent_str + '    \\n',
        indent_str + '    # Store job in Redis\\n',
        indent_str + '    try:\\n',
        indent_str + '        get_redis_client().hset(f\"job:{job_id}\", mapping={}\\n',
        indent_str + '            \"owner\": owner,\\n',
        indent_str + '            \"repo\": repo,\\n',
        indent_str + '            \"threshold\": threshold,\\n',
        indent_str + '            \"status\": \"queued\",\\n',
        indent_str + '            \"created_at\": datetime.utcnow().isoformat()\\n',
        indent_str + '        })\\n',
        indent_str + '    except redis.exceptions.ConnectionError as e:\\n',
        indent_str + '        raise HTTPException(status_code=503, detail=\"Redis service unavailable\")\\n',
        indent_str + '    \\n',
        indent_str + '    # Trigger background processing\\n',
        indent_str + '    # In production: push to Celery queue\\n',
        indent_str + '    # For now: process synchronously with job tracking\\n',
        indent_str + '    result = await analyze_repo(owner, repo, threshold)\\n',
        indent_str + '    \\n',
        indent_str + '    # Store result\\n',
        indent_str + '    try:\\n',
        indent_str + '        get_redis_client().setex(f\"repo:{owner}:{repo}\", 86400, json.dumps(result))\\n',
        indent_str + '        get_redis_client().hset(f\"job:{job_id}\", \"status\", \"completed\")\\n',
        indent_str + '        get_redis_client().hset(f\"job:{job_id}\", \"result_key\", f\"repo:{owner}:{repo}\")\\n',
        indent_str + '    except redis.exceptions.ConnectionError as e:\\n',
        indent_str + '        raise HTTPException(status_code=503, detail=\"Redis service unavailable\")\\n',
        indent_str + '    \\n',
        indent_str + '    return {\"job_id\": job_id, \"status\": \"completed\", \"result_key\": f\"repo:{owner}:{repo}\"}\\n'
    ]
    # Note: we need to ensure we don't double indent; the original lines already had the correct indentation.
    # We'll replace from post_idx to analyze_idx.
    lines = lines[:post_idx] + new_submit + lines[analyze_idx:]
    
    # 3. Replace get_job_status function
    # Find '@app.get("/api/jobs/{job_id}")'
    getjob_idx = None
    for i, line in enumerate(lines):
        if line.strip() == '@app.get("/api/jobs/{job_id}")':
            getjob_idx = i
            break
    if getjob_idx is None:
        print("Could not find get_job_status")
        sys.exit(1)
    # Find the end of the function: look for the next line that starts with '@app' or 'async def' after getjob_idx.
    # The next function is get_repo_score.
    getscore_idx = None
    for i in range(getjob_idx+1, len(lines)):
        if lines[i].strip().startswith('@app.get("/api/score/{owner}/{repo}")'):
            getscore_idx = i
            break
    if getscore_idx is None:
        print("Could not find get_repo_score")
        sys.exit(1)
    # We'll replace lines[getjob_idx:getscore_idx] with new get_job_status.
    # Keep the decorator line, def line, docstring.
    # Get indent of function body.
    body_indent2 = len(lines[getjob_idx+3]) - len(lines[getjob_idx+3].lstrip())
    indent_str2 = lines[getjob_idx+3][:body_indent2]
    
    new_getjob = [
        lines[getjob_idx],  # '@app.get("/api/jobs/{job_id}")\\n'
        lines[getjob_idx+1], # 'async def get_job_status(job_id: str):\\n'
        lines[getjob_idx+2], # '    """Get the status of a submitted job."""\\n'
        indent_str2 + '    try:\\n',
        indent_str2 + '        job_data = get_redis_client().hgetall(f\"job:{job_id}\")\\n',
        indent_str2 + '    except redis.exceptions.ConnectionError as e:\\n',
        indent_str2 + '        raise HTTPException(status_code=503, detail=\"Redis service unavailable\")\\n',
        indent_str2 + '    \\n',
        indent_str2 + '    if not job_data:\\n',
        indent_str2 + '        raise HTTPException(status_code=404, detail=\"Job not found\")\\n',
        indent_str2 + '    \\n',
        indent_str2 + '    result_key = job_data.get(\"result_key\")\\n',
        indent_str2 + '    if result_key:\\n',
        indent_str2 + '        try:\\n',
        indent_str2 + '            result_data = get_redis_client().get(result_key)\\n',
        indent_str2 + '        except redis.exceptions.ConnectionError as e:\\n',
        indent_str2 + '            raise HTTPException(status_code=503, detail=\"Redis service unavailable\")\\n',
        indent_str2 + '        if result_data:\\n',
        indent_str2 + '            job_data[\"result\"] = json.loads(result_data)\\n',
        indent_str2 + '    \\n',
        indent_str2 + '    return job_data\\n'
    ]
    # Replace
    lines = lines[:getjob_idx] + new_getjob + lines[getscore_idx:]
    
    # Write back
    with open(filename, 'w', newline='') as f:
        f.writelines(lines)
    print("Patched api_server.py successfully")

if __name__ == '__main__':
    main()
