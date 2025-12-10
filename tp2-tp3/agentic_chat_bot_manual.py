from agents import create_agentic_rag_system


if __name__ == "__main__":

    print("=" * 60)
    print("AGENTIC RAG SYSTEM - USING LANGCHAIN AGENTS")
    print("=" * 60)
    
    # Create the agentic RAG system
    print("\n🔧 Initializing agentic RAG system...")
    agent_executor, pinecone_client, embedding_model = create_agentic_rag_system()
    
    # List available indexes
    print("\n📚 Available indexes:")
    try:
        for idx in pinecone_client.list_indexes():
            print(f"  - {idx.name}")
    except Exception as e:
        print(f"  Error listing indexes: {e}")
    
    print("\n" + "=" * 60)
    
    # Test queries
    test_queries = [
        "What is the work experience?",  # No name - uses default
        "What was Maxim Dorogov's work experience?",  # Specific person
        "Tell me about John Doe's education",  # Another person (may not exist)
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 60}")
        print(f"Query {i}: {query}")
        print("-" * 60)
        
        try:
            # Invoke the agent with the query
            result = agent_executor.invoke({
                "input": query,
                "pinecone_client": pinecone_client,
                "embedding_model": embedding_model
            })
            
            print(f"\n🤖 Agent Response:")
            print(result["output"])
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)
    
    # Interactive mode
    print("\n💬 Enter interactive mode (type 'quit' to exit)")
    print("-" * 60)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            result = agent_executor.invoke({
                "input": user_input,
                "pinecone_client": pinecone_client,
                "embedding_model": embedding_model
            })
            
            print(f"\n🤖 Assistant: {result['output']}\n")
            
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
