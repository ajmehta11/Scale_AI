#!/bin/bash

# Comprehensive test script for all index types
# Tests: Create libraries, documents, chunks, and search across all indexes

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' 

INDEXES=("brute_force" "kd_tree" "lsh" "ivfpq")
BASE_PORT=8000
IMAGE_NAME="vector-db"
API_KEY="${COHERE_API_KEY}"

NUM_LIBRARIES=3
NUM_DOCUMENTS_PER_LIBRARY=5
NUM_CHUNKS_PER_DOCUMENT=20
TOTAL_CHUNKS=$((NUM_LIBRARIES * NUM_DOCUMENTS_PER_LIBRARY * NUM_CHUNKS_PER_DOCUMENT))

if [ -z "$API_KEY" ]; then
    echo -e "${RED}Error: COHERE_API_KEY not set${NC}"
    echo "Please set it: export COHERE_API_KEY=your-api-key"
    exit 1
fi

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    COMPREHENSIVE INDEX TESTING                               ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Test Configuration:${NC}"
echo "  Libraries: $NUM_LIBRARIES"
echo "  Documents per library: $NUM_DOCUMENTS_PER_LIBRARY"
echo "  Chunks per document: $NUM_CHUNKS_PER_DOCUMENT"
echo "  Total chunks: $TOTAL_CHUNKS"
echo ""

CHUNK_TEXTS=(
    "Machine learning is a subset of artificial intelligence that enables computers to learn from data"
    "Deep learning uses neural networks with multiple layers to process complex patterns"
    "Natural language processing helps computers understand and generate human language"
    "Computer vision enables machines to interpret and understand visual information from images"
    "Reinforcement learning trains agents through reward and penalty signals"
    "Supervised learning uses labeled data to train predictive models"
    "Unsupervised learning finds patterns in unlabeled data without explicit guidance"
    "Transfer learning applies knowledge from one domain to another related domain"
    "Neural networks are inspired by biological neurons in the human brain"
    "Convolutional neural networks excel at processing grid-like data such as images"
    "Recurrent neural networks are designed to handle sequential data and time series"
    "Transformers revolutionized NLP with self-attention mechanisms"
    "Word embeddings represent words as dense vectors in continuous space"
    "Gradient descent optimizes model parameters by minimizing loss functions"
    "Backpropagation computes gradients for updating neural network weights"
    "Overfitting occurs when models memorize training data instead of generalizing"
    "Regularization techniques prevent overfitting by constraining model complexity"
    "Cross-validation assesses model performance on different data splits"
    "Feature engineering transforms raw data into meaningful representations"
    "Dimensionality reduction techniques compress high-dimensional data"
)

wait_for_health() {
    local port=$1
    local max_attempts=30
    local attempt=1

    echo "Waiting for API to be ready on port $port..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "http://localhost:$port/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ API is ready${NC}"
            return 0
        fi
        sleep 2
        attempt=$((attempt + 1))
    done

    echo -e "${RED}✗ API failed to start${NC}"
    return 1
}

test_index() {
    local index=$1
    local port=$2
    local container_name="test-index-$index"

    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Testing Index: ${index}${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo ""

    echo -e "${YELLOW}Step 1: Starting container${NC}"
    docker run -d \
        --name "$container_name" \
        -p "$port:8000" \
        -e COHERE_API_KEY="$API_KEY" \
        -e INDEX_TYPE="$index" \
        "$IMAGE_NAME" > /dev/null

    if ! wait_for_health "$port"; then
        docker logs "$container_name"
        docker stop "$container_name" 2>/dev/null || true
        docker rm "$container_name" 2>/dev/null || true
        return 1
    fi

    START_TIME=$(date +%s)

    echo ""
    echo -e "${YELLOW}Step 2: Creating $NUM_LIBRARIES libraries${NC}"
    LIBRARY_IDS=()
    for i in $(seq 1 $NUM_LIBRARIES); do
        LIB_RESPONSE=$(curl -s -X POST "http://localhost:$port/libraries" \
            -H "Content-Type: application/json" \
            -d "{\"name\": \"Library $i - $index\", \"metadata\": {\"category\": \"category$((i % 3))\", \"index\": \"$index\"}}")

        LIB_ID=$(echo $LIB_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])")
        LIBRARY_IDS+=("$LIB_ID")
        echo "  ✓ Created library $i: $LIB_ID"
    done

    echo ""
    echo -e "${YELLOW}Step 3: Creating documents and chunks${NC}"
    DOCUMENT_IDS=()
    CHUNK_COUNT=0

    for lib_idx in "${!LIBRARY_IDS[@]}"; do
        LIB_ID="${LIBRARY_IDS[$lib_idx]}"
        echo "  Library $((lib_idx + 1)):"

        for doc_num in $(seq 1 $NUM_DOCUMENTS_PER_LIBRARY); do
            DOC_RESPONSE=$(curl -s -X POST "http://localhost:$port/documents" \
                -H "Content-Type: application/json" \
                -d "{\"name\": \"Document $doc_num\", \"library_id\": \"$LIB_ID\", \"metadata\": {\"topic\": \"topic$((doc_num % 5))\"}}")

            DOC_ID=$(echo $DOC_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])")
            DOCUMENT_IDS+=("$DOC_ID")

            for chunk_num in $(seq 1 $NUM_CHUNKS_PER_DOCUMENT); do
                text_idx=$(( (CHUNK_COUNT % ${#CHUNK_TEXTS[@]}) ))
                CHUNK_TEXT="${CHUNK_TEXTS[$text_idx]} (chunk $CHUNK_COUNT)"

                curl -s -X POST "http://localhost:$port/chunks" \
                    -H "Content-Type: application/json" \
                    -d "{\"text\": \"$CHUNK_TEXT\", \"document_id\": \"$DOC_ID\", \"metadata\": {\"chunk_num\": $chunk_num, \"doc_num\": $doc_num}}" \
                    > /dev/null

                CHUNK_COUNT=$((CHUNK_COUNT + 1))
            done

            echo "    ✓ Document $doc_num: $NUM_CHUNKS_PER_DOCUMENT chunks"
        done
    done

    LOAD_TIME=$(($(date +%s) - START_TIME))
    echo ""
    echo -e "${GREEN}  Total chunks created: $CHUNK_COUNT${NC}"
    echo -e "${GREEN}  Load time: ${LOAD_TIME}s${NC}"

    # Verify stats
    echo ""
    echo -e "${YELLOW}Step 4: Verifying database statistics${NC}"
    STATS=$(curl -s "http://localhost:$port/stats")
    echo "  $STATS" | python3 -m json.tool

    NUM_CHUNKS=$(echo $STATS | python3 -c "import sys, json; print(json.load(sys.stdin)['num_chunks'])")

    if [ "$NUM_CHUNKS" -eq "$TOTAL_CHUNKS" ]; then
        echo -e "${GREEN}  ✓ Chunk count verified: $NUM_CHUNKS${NC}"
    else
        echo -e "${RED}  ✗ Chunk count mismatch: expected $TOTAL_CHUNKS, got $NUM_CHUNKS${NC}"
    fi

    # Perform searches
    echo ""
    echo -e "${YELLOW}Step 5: Testing search functionality${NC}"

    SEARCH_QUERIES=(
        "machine learning and artificial intelligence"
        "neural networks and deep learning"
        "natural language processing"
        "computer vision and image recognition"
        "supervised and unsupervised learning"
    )

    SEARCH_START=$(date +%s)
    TOTAL_SEARCH_TIME=0

    for query in "${SEARCH_QUERIES[@]}"; do
        QUERY_START=$(python3 -c 'import time; print(int(time.time() * 1000))')

        SEARCH_RESULT=$(curl -s -X POST "http://localhost:$port/search" \
            -H "Content-Type: application/json" \
            -d "{\"query\": \"$query\", \"k\": 5}")

        QUERY_END=$(python3 -c 'import time; print(int(time.time() * 1000))')
        QUERY_TIME=$((QUERY_END - QUERY_START))
        TOTAL_SEARCH_TIME=$((TOTAL_SEARCH_TIME + QUERY_TIME))

        RESULT_COUNT=$(echo $SEARCH_RESULT | python3 -c "import sys, json; print(json.load(sys.stdin)['count'])")
        TOP_SCORE=$(echo $SEARCH_RESULT | python3 -c "import sys, json; r=json.load(sys.stdin)['results']; print(f\"{r[0]['score']:.4f}\" if r else '0')")

        echo "  Query: \"${query:0:40}...\""
        echo "    Results: $RESULT_COUNT, Top score: $TOP_SCORE, Latency: ${QUERY_TIME}ms"
    done

    AVG_SEARCH_TIME=$((TOTAL_SEARCH_TIME / ${#SEARCH_QUERIES[@]}))
    echo ""
    echo -e "${GREEN}  ✓ All searches completed${NC}"
    echo -e "${GREEN}  Average search latency: ${AVG_SEARCH_TIME}ms${NC}"

    # Test metadata filtering
    echo ""
    echo -e "${YELLOW}Step 6: Testing metadata filtering${NC}"

    FILTER_RESULT=$(curl -s -X POST "http://localhost:$port/search" \
        -H "Content-Type: application/json" \
        -d '{"query": "machine learning", "k": 10, "metadata_filter": {"chunk_num": 1}}')

    FILTER_COUNT=$(echo $FILTER_RESULT | python3 -c "import sys, json; print(json.load(sys.stdin)['count'])")
    echo "  Search with metadata filter (chunk_num=1): $FILTER_COUNT results"

    # Test library filtering
    FIRST_LIB="${LIBRARY_IDS[0]}"
    LIB_FILTER_RESULT=$(curl -s -X POST "http://localhost:$port/search" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"learning\", \"k\": 20, \"library_id\": \"$FIRST_LIB\"}")

    LIB_FILTER_COUNT=$(echo $LIB_FILTER_RESULT | python3 -c "import sys, json; print(json.load(sys.stdin)['count'])")
    echo "  Search with library filter: $LIB_FILTER_COUNT results"

    # Performance summary
    TOTAL_TIME=$(($(date +%s) - START_TIME))

    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Index: ${index} - Test Summary${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  ✓ Libraries created: $NUM_LIBRARIES${NC}"
    echo -e "${GREEN}  ✓ Documents created: ${#DOCUMENT_IDS[@]}${NC}"
    echo -e "${GREEN}  ✓ Chunks created: $CHUNK_COUNT${NC}"
    echo -e "${GREEN}  ✓ Searches completed: ${#SEARCH_QUERIES[@]}${NC}"
    echo -e "${GREEN}  ✓ Average search latency: ${AVG_SEARCH_TIME}ms${NC}"
    echo -e "${GREEN}  ✓ Total test time: ${TOTAL_TIME}s${NC}"
    echo -e "${GREEN}  ✓ Throughput: $((CHUNK_COUNT / TOTAL_TIME)) chunks/sec${NC}"
    echo ""

    # Cleanup
    echo "Stopping and removing container..."
    docker stop "$container_name" > /dev/null 2>&1
    docker rm "$container_name" > /dev/null 2>&1

    # Store results for comparison
    echo "$index,$CHUNK_COUNT,$TOTAL_TIME,$AVG_SEARCH_TIME" >> /tmp/index_test_results.csv
}

# Main execution
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -t "$IMAGE_NAME" . > /dev/null 2>&1
echo -e "${GREEN}✓ Image built${NC}"

# Clear previous results
rm -f /tmp/index_test_results.csv
echo "index,chunks,load_time_sec,avg_search_ms" > /tmp/index_test_results.csv

# Test each index
for i in "${!INDEXES[@]}"; do
    index="${INDEXES[$i]}"
    port=$((BASE_PORT + i))

    if test_index "$index" "$port"; then
        echo -e "${GREEN}✓ $index tests passed${NC}"
    else
        echo -e "${RED}✗ $index tests failed${NC}"
    fi

    sleep 2
done

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                         PERFORMANCE COMPARISON                                ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

python3 << 'EOF'
import csv

print(f"{'Index':<15} {'Chunks':<10} {'Load Time':<15} {'Avg Search':<15} {'Throughput':<15}")
print("=" * 75)

with open('/tmp/index_test_results.csv', 'r') as f:
    reader = csv.DictReader(f)
    results = list(reader)

    if not results:
        print("No results to display")
    else:
        for row in results:
            index = row['index']
            chunks = row['chunks']
            load_time = float(row['load_time_sec'])
            avg_search = row['avg_search_ms']
            throughput = int(chunks) / load_time if load_time > 0 else 0

            print(f"{index:<15} {chunks:<10} {load_time:<15.1f}s {avg_search:<15}ms {throughput:<15.1f} c/s")

        print()

        # Find fastest
        if len(results) > 0:
            fastest_load = min(results, key=lambda x: float(x['load_time_sec']))
            fastest_search = min(results, key=lambda x: float(x['avg_search_ms']))

            print(f"🏆 Fastest loading:  {fastest_load['index']} ({fastest_load['load_time_sec']}s)")
            print(f"🏆 Fastest search:   {fastest_search['index']} ({fastest_search['avg_search_ms']}ms)")
EOF

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                         ALL TESTS COMPLETED                                   ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Detailed results saved to: /tmp/index_test_results.csv"
