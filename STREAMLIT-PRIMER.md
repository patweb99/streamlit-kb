# Streamlit Primer: Build Web Apps with Python

## What is Streamlit?

Streamlit turns Python scripts into beautiful web apps in minutes. No need to learn HTML, CSS, or JavaScript - just write Python and get a web app!

Visit [streamlit.io](https://streamlit.io) to see examples and learn more.

## Quick Start

### 1. Install Streamlit
```bash
pip install streamlit
```

### 2. Create Your First App
Make a file called `app.py`:

```python
import streamlit as st

st.title("My First App")
st.write("Hello, world!")
```

### 3. Run Your App
```bash
streamlit run app.py
```

Your app opens in your browser at `http://localhost:8501` 🎉

## The Basics

### Display Text
```python
import streamlit as st

st.title("Big Title")
st.header("Header")
st.write("Any text or data")
st.markdown("**Bold** and *italic* text")
```

### Get User Input
```python
name = st.text_input("What's your name?")
age = st.number_input("How old are you?", min_value=0)

if st.button("Say Hello"):
    st.write(f"Hello {name}! You are {age} years old.")
```

### Show Data
```python
import pandas as pd

# Create some data
data = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Score': [85, 90, 78]
})

# Display it
st.dataframe(data)
st.bar_chart(data.set_index('Name'))
```

## Layout Your App

### Use the Sidebar
```python
st.sidebar.title("Menu")
choice = st.sidebar.selectbox("Pick one:", ["Option A", "Option B"])

if choice == "Option A":
    st.write("You picked A!")
else:
    st.write("You picked B!")
```

### Create Columns
```python
col1, col2 = st.columns(2)

col1.write("Left column")
col2.write("Right column")
```

### Use Tabs
```python
tab1, tab2 = st.tabs(["Home", "About"])

with tab1:
    st.write("Welcome home!")

with tab2:
    st.write("About this app...")
```

## Simple Example: To-Do App

```python
import streamlit as st

st.title("📝 Simple To-Do App")

# Initialize the todo list in session state
if 'todos' not in st.session_state:
    st.session_state.todos = []

# Add new todo
new_todo = st.text_input("Add a new task:")
if st.button("Add") and new_todo:
    st.session_state.todos.append(new_todo)
    st.success(f"Added: {new_todo}")

# Show current todos
if st.session_state.todos:
    st.subheader("Your Tasks:")
    for i, todo in enumerate(st.session_state.todos):
        col1, col2 = st.columns([4, 1])
        col1.write(f"{i+1}. {todo}")
        if col2.button("✓", key=f"done_{i}"):
            st.session_state.todos.pop(i)
            st.rerun()
else:
    st.write("No tasks yet. Add one above!")
```

## Useful Features

### File Upload
```python
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.dataframe(data)
```

### Progress and Status
```python
import time

if st.button("Start Process"):
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)
    st.success("Done!")
```

### Cache Expensive Operations
```python
@st.cache_data
def load_big_data():
    # This only runs once, then remembers the result
    return pd.read_csv("huge_file.csv")

data = load_big_data()  # Fast after first run!
```

## Tips for Success

1. **Start simple** - Build basic functionality first
2. **Use st.write()** - It displays almost anything
3. **Check session state** - Use `st.session_state` to remember things
4. **Test often** - Streamlit auto-refreshes when you save

## Deploy Your App

### Free with Streamlit Community Cloud
1. Put your code on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub and deploy
4. Share your app with the world!

## Learn More

- **Streamlit Website**: [streamlit.io](https://streamlit.io)
- **Documentation**: [docs.streamlit.io](https://docs.streamlit.io)
- **Gallery**: [streamlit.io/gallery](https://streamlit.io/gallery) - See what others built
- **Community**: [discuss.streamlit.io](https://discuss.streamlit.io) - Get help

## Common Widgets Cheat Sheet

```python
# Text inputs
text = st.text_input("Enter text:")
number = st.number_input("Enter number:")
password = st.text_input("Password:", type="password")

# Selections
option = st.selectbox("Choose one:", ["A", "B", "C"])
options = st.multiselect("Choose many:", ["A", "B", "C"])
choice = st.radio("Pick one:", ["Yes", "No"])

# Interactive
clicked = st.button("Click me!")
checked = st.checkbox("Check me!")
value = st.slider("Slide me:", 0, 100, 50)

# File and date
file = st.file_uploader("Upload file:")
date = st.date_input("Pick date:")
color = st.color_picker("Pick color:")
```

---

**Ready to build?** Start with the simple examples above, then visit [streamlit.io](https://streamlit.io) for more inspiration! 🚀