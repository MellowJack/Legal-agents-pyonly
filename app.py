import streamlit as st
from crew_logic import crew

st.set_page_config(page_title="Legal Argument Assistant", layout="centered")
st.title("⚖️ Legal Argument Matcher")

argument = st.text_area("📝 Enter your legal argument", height=200)

if st.button("Run AI Analysis"):
    if not argument.strip():
        st.warning("Please enter a legal argument.")
    else:
        with st.spinner("Analyzing your argument..."):
            result = crew.kickoff(inputs={"argument": argument})
            st.success("Done!")

            st.subheader("🧾 Legal Analysis")
            # ✅ Display result in a scrollable textbox instead of markdown
            st.text_area("Result", value=result, height=300)