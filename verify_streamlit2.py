import requests

# Check Streamlit stream for app content - get more lines
r = requests.get('http://localhost:8501/_stcore/stream', stream=True, timeout=30)
count = 0
for line in r.iter_lines():
    text = line.decode()
    print(f'Line {count}: {text[:200]}')
    count += 1
    if count > 50:
        break

print("Done")