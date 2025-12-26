#!/bin/bash
# Build script for Self-Driving RL System

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo "============================================================"
    echo -e "${GREEN}$1${NC}"
    echo "============================================================"
}

print_warning() {
    echo -e "${YELLOW}Warning: $1${NC}"
}

print_error() {
    echo -e "${RED}Error: $1${NC}"
}

# Check prerequisites
check_prereqs() {
    print_header "Checking prerequisites..."
    
    # Python
    if command -v python3 &> /dev/null; then
        echo "Python: $(python3 --version)"
    else
        print_error "Python 3 not found"
        exit 1
    fi
    
    # CMake
    if command -v cmake &> /dev/null; then
        echo "CMake: $(cmake --version | head -n1)"
    else
        print_error "CMake not found"
        exit 1
    fi
    
    # Java
    if command -v java &> /dev/null; then
        echo "Java: $(java --version 2>&1 | head -n1)"
    else
        print_warning "Java not found - Java components will be skipped"
    fi
    
    # Gradle
    if command -v gradle &> /dev/null; then
        echo "Gradle: $(gradle --version 2>&1 | grep Gradle | head -n1)"
    fi
}

# Install Python dependencies
install_python_deps() {
    print_header "Installing Python dependencies..."
    
    cd "$PROJECT_ROOT/python"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate and install
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    
    echo "Python dependencies installed successfully!"
}

# Build C++ engine
build_cpp() {
    print_header "Building C++ Inference Engine..."
    
    cd "$PROJECT_ROOT/cpp"
    
    # Create build directory
    mkdir -p build
    cd build
    
    # Configure CMake
    CMAKE_ARGS=""
    
    if [ -n "$ONNXRUNTIME_ROOT" ]; then
        CMAKE_ARGS="$CMAKE_ARGS -DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT"
    else
        print_warning "ONNXRUNTIME_ROOT not set. CMake will try to find ONNX Runtime automatically."
        print_warning "Set ONNXRUNTIME_ROOT=/path/to/onnxruntime for best results."
    fi
    
    if [ "$USE_CUDA" = "1" ]; then
        CMAKE_ARGS="$CMAKE_ARGS -DUSE_CUDA=ON"
    fi
    
    cmake .. $CMAKE_ARGS
    
    # Build
    make -j$(nproc)
    
    echo ""
    echo "C++ build complete!"
    echo "Libraries: $PROJECT_ROOT/cpp/build/lib"
    echo "Binaries: $PROJECT_ROOT/cpp/build/bin"
}

# Build Java server
build_java() {
    print_header "Building Java Server..."
    
    cd "$PROJECT_ROOT/java"
    
    # Check for Gradle wrapper
    if [ -f "gradlew" ]; then
        ./gradlew build -x test
    elif command -v gradle &> /dev/null; then
        gradle build -x test
    else
        print_warning "Gradle not found - skipping Java build"
        return
    fi
    
    echo ""
    echo "Java build complete!"
    echo "JAR: $PROJECT_ROOT/java/build/libs/"
}

# Create Gradle wrapper
create_gradle_wrapper() {
    print_header "Creating Gradle wrapper..."
    
    cd "$PROJECT_ROOT/java"
    
    if command -v gradle &> /dev/null; then
        gradle wrapper
        echo "Gradle wrapper created!"
    else
        print_warning "Gradle not found - cannot create wrapper"
    fi
}

# Full build
build_all() {
    check_prereqs
    install_python_deps
    build_cpp
    build_java
    
    print_header "Build Complete!"
    echo ""
    echo "Next steps:"
    echo "1. Train a model:"
    echo "   cd python && source venv/bin/activate"
    echo "   python train_self_driving.py --total-timesteps 500000"
    echo ""
    echo "2. Export to ONNX:"
    echo "   python export_onnx.py --checkpoint logs/*/checkpoints/model_final.pt"
    echo ""
    echo "3. Test C++ engine:"
    echo "   ./cpp/build/bin/test_engine models/self_driving_policy.onnx"
    echo ""
    echo "4. Run Java server:"
    echo "   cd java && ./gradlew bootRun"
}

# Clean build artifacts
clean() {
    print_header "Cleaning build artifacts..."
    
    rm -rf "$PROJECT_ROOT/cpp/build"
    rm -rf "$PROJECT_ROOT/java/build"
    rm -rf "$PROJECT_ROOT/java/.gradle"
    rm -rf "$PROJECT_ROOT/python/venv"
    rm -rf "$PROJECT_ROOT/python/__pycache__"
    rm -rf "$PROJECT_ROOT/python/rl/__pycache__"
    
    echo "Clean complete!"
}

# Show help
show_help() {
    echo "Self-Driving RL System Build Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  all       Build everything (default)"
    echo "  python    Install Python dependencies"
    echo "  cpp       Build C++ engine"
    echo "  java      Build Java server"
    echo "  wrapper   Create Gradle wrapper"
    echo "  clean     Clean build artifacts"
    echo "  help      Show this help"
    echo ""
    echo "Environment variables:"
    echo "  ONNXRUNTIME_ROOT   Path to ONNX Runtime installation"
    echo "  USE_CUDA=1         Enable CUDA support in C++ build"
}

# Main
case "${1:-all}" in
    all)
        build_all
        ;;
    python)
        install_python_deps
        ;;
    cpp)
        build_cpp
        ;;
    java)
        build_java
        ;;
    wrapper)
        create_gradle_wrapper
        ;;
    clean)
        clean
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
