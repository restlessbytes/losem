CREATE_EMBEDDINGS_TABLE = """
CREATE TABLE IF NOT EXISTS embeddings ( 
document_id VARCHAR, 
id VARCHAR, 
enumeration INTEGER,
content VARCHAR, 
filename VARCHAR, 
filepath VARCHAR, 
data_type VARCHAR, 
vector FLOAT[1024],  
PRIMARY KEY (document_id, id, enumeration));
"""

CREATE_HNSW_INDEX = """INSTALL vss;
LOAD vss;
SET hnsw_enable_experimental_persistence = true;
CREATE INDEX cosine_index ON embeddings USING HNSW (vector)
WITH (metric = 'cosine');
"""

CREATE_EMBEDDING_MODELS_TABLE = """
CREATE TABLE IF NOT EXISTS document_embedding_model (
model_name VARCHAR PRIMARY KEY
);
"""

INSERT_CHUNKS = """
INSERT OR REPLACE 
INTO embeddings
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

# There can only be ONE embedding model per database
INSERT_EMBEDDING_MODEL = "INSERT INTO document_embedding_model VALUES (?);"

TEXT_CHUNK_COLUMNS = (
    "document_id, id, enumeration, content, filename, filepath, data_type"
)
SELECT_TEXT_CHUNKS = f"SELECT {TEXT_CHUNK_COLUMNS} FROM embeddings"  # nosec
SELECT_EMBEDDINGS = f"SELECT {TEXT_CHUNK_COLUMNS}, vector FROM embeddings"  # nosec
SELECT_MODEL_NAME = "SELECT model_name FROM document_embedding_model LIMIT 1"  # nosec
