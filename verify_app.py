import requests

# Check if Streamlit app is running and rendering
r = requests.get('http://localhost:8501', timeout=5)
print("Status:", r.status_code)
print("Contains 'RepoScore':", 'RepoScore' in r.text)
print("Contains 'Predict':", 'Predict' in r.text)
print("Contains 'Quality':", 'Quality' in r.text)

# Check the stream for actual app content
r2 = requests.get('http://localhost:8501/_stcore/stream', stream=True, timeout=10)
for i, line in enumerate(r2.iter_lines()):
    text = line.decode()
    if 'RepoScore' in text or 'Predict' in text or 'AI Review' in text or 'quality' in text.lower():
        print(f"Line {i}: {text[:300]}")
    if i > 20:
        break

print("\nDone")