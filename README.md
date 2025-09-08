#Steps for backend
files that are created

##- pinecone-setup.py

-   first to be created is pinecone python file so that we can create indexing on the pinecone to store the vectors in there

## ingest.py

-   this is a first time setup file to create chunks using local files

-This pipeline transforms raw documents into an optimized format for AI-powered search and answering - the foundation of any RAG system!

-   this code create vectors on pinecode and reference to this vectors in firestore so that
    When a user asks a question:

        <li>Pinecone finds similar vectors ("chunk-23", "chunk-87")</li>
        <li>We use these IDs to fetch actual text from Firestore</li>
        <li>The text is then fed to the AI to generate answers</li>

## incremental_ingest.py

-   this is created so that you can add new files to the pinecone and firebase, it also updates already ingested files.
