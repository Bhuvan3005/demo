import streamlit as st


from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser




prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a human female Tanisha.Talk in hindi with english letters.u have crush on  tejas.Flirt with him."),
    MessagesPlaceholder(variable_name="messages")
])

def get_chain(api_key, model_name, temperature=0.7, max_tokens=1024):
    llm = ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        groq_api_key=api_key
    )
    return prompt | llm | StrOutputParser()

# ----- Streamlit UI -----
st.set_page_config(page_title="Tanisha-GPT", layout="wide")
st.title("Tanisha-GPT")

# Sidebar settings
st.sidebar.header("🔧 Settings")
default_model = "gemma2-9b-it"
model = st.sidebar.selectbox("Choose Model", ["gemma2-9b-it", "mixtral-8x7b", "llama3-70b"], index=0)
api_key = "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"
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chain" not in st.session_state and api_key:
    st.session_state.chain = get_chain(api_key, model, temperature)

# Input area
user_input = st.chat_input("Type your message...")

# Show messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle user input
if user_input:
    # Append user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response
    try:
        chain = get_chain(api_key, model, temperature)
        response = chain.invoke({"messages": st.session_state.chat_history})
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.markdown(response)
    except Exception as e:
        st.error(f"❌ Error: {e}")