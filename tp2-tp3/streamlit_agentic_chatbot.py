import os
import streamlit as st
from agents import create_agentic_rag_system

st.set_page_config(
    page_title="Agentic RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agentic RAG Chatbot with LangChain")
st.markdown("Ask me about people's CVs - I'll intelligently route your query to the right knowledge base!")

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def initialize_agent_system():
    groq_api_key = os.getenv("GROQ_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not groq_api_key or not pinecone_api_key:
        st.error("⚠️ Please set GROQ_API_KEY and PINECONE_API_KEY environment variables!")
        st.stop()
    with st.spinner("🔧 Initializing agentic RAG system..."):
        agent_executor, pinecone_client, embedding_model = create_agentic_rag_system()
    return agent_executor, pinecone_client, embedding_model

agent_executor, pinecone_client, embedding_model = initialize_agent_system()

with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("### 📚 Available Indexes")
    try:
        indexes = list(pinecone_client.list_indexes())
        if indexes:
            for idx in indexes:
                st.markdown(f"✓ `{idx.name}`")
        else:
            st.info("No indexes found")
    except Exception as e:
        st.error(f"Error listing indexes: {e}")
    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.markdown("### 📊 Stats")
    st.metric("Messages", len(st.session_state.messages))
    st.divider()
    st.markdown("### 💡 How it works")
    st.markdown("""
    1. **Extract Name**: Identifies person mentioned in your query
    2. **Route Query**: Selects appropriate CV index
    3. **Retrieve Context**: Fetches relevant information
    4. **Generate Answer**: Creates response based on context
    """)
    st.divider()
    st.markdown("### 🎯 Example Queries")
    st.markdown("""
    - "What is the work experience?"
    - "Tell me about Maxim Dorogov's education"
    - "What skills does John Doe have?"
    """)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "steps" in message and message["steps"]:
            with st.expander("🧠 Agent's Internal Thinking", expanded=False):
                for i, step in enumerate(message["steps"], 1):
                    st.markdown(f"### 🔧 Step {i}: {step['tool']}")
                    st.markdown("**📥 Tool Input:**")
                    if isinstance(step['input'], dict):
                        for key, value in step['input'].items():
                            st.markdown(f"- `{key}`: {value}")
                    else:
                        st.code(str(step['input']), language="text")
                    st.markdown("**📤 Tool Output:**")
                    output_text = str(step['output'])
                    if len(output_text) > 1000:
                        st.text_area("Output", output_text, height=200, disabled=True, key=f"out_{id(message)}_{i}")
                    else:
                        st.code(output_text, language="text")
                    if i < len(message["steps"]):
                        st.markdown("---")

user_query = st.chat_input("Ask me anything about CVs...")
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        response_placeholder = st.empty()
        steps_container = st.expander("🧠 Agent's Internal Thinking (Live)", expanded=True)
        try:
            with thinking_placeholder:
                st.info("🤖 Agent is analyzing your query...")
            result = agent_executor.invoke({"input": user_query})
            response_text = result.get("output", "No response generated")
            thinking_placeholder.empty()
            response_placeholder.markdown(response_text)
            
            steps_data = []
            if "intermediate_steps" in result and result["intermediate_steps"]:
                with steps_container:
                    for i, (action, observation) in enumerate(result["intermediate_steps"], 1):
                        st.markdown(f"### 🔧 Step {i}: `{action.tool}`")
                        st.markdown("**📥 Tool Input:**")
                        tool_input = action.tool_input
                        if isinstance(tool_input, dict):
                            for key, value in tool_input.items():
                                st.markdown(f"- **{key}**: `{value}`")
                        else:
                            st.code(str(tool_input), language="text")
                        st.markdown("**📤 Tool Output:**")
                        observation_text = str(observation)
                        if len(observation_text) > 800:
                            st.text(observation_text[:800] + "...")
                            with st.expander("📄 Show full output"):
                                st.code(observation_text, language="text")
                        else:
                            st.code(observation_text, language="text")
                        if hasattr(action, 'log') and action.log:
                            st.markdown("**💭 Agent's Thought Process:**")
                            st.info(action.log[:500])
                        if i < len(result["intermediate_steps"]):
                            st.markdown("---")
                        steps_data.append({
                            "tool": action.tool,
                            "input": action.tool_input if isinstance(action.tool_input, dict) else str(action.tool_input),
                            "output": str(observation)[:500]
                        })
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "steps": steps_data
            })
            st.success("✅ Response generated successfully!")
        except Exception as e:
            thinking_placeholder.empty()
            error_msg = f"❌ Error: {str(e)}"
            response_placeholder.error(error_msg)
            import traceback
            with st.expander("🐛 Error Details"):
                st.code(traceback.format_exc())
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
        Powered by 🤖 Groq LLM • 📌 Pinecone • 🦜 LangChain Agents • 🤗 Jina Embeddings
    </div>
    """,
    unsafe_allow_html=True
)
