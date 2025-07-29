from app.rag_chain import rag_pipeline  # Import the pipeline instance

if __name__ == "__main__":
    while True:
        user_query = input("Ask a question (or 'exit'): ")
        if user_query.lower() == 'exit':
            break
        response = rag_pipeline.query(user_query)  # Use the query method
        print("\nAnswer:\n", response)