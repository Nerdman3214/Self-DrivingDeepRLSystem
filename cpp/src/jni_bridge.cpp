/**
 * JNI Bridge Implementation
 * 
 * Bridges Java calls to the C++ inference engine.
 */

#include "jni_bridge.h"
#include "inference_engine.h"

#include <iostream>
#include <string>
#include <memory>

using namespace selfdriving;

// Global to track last inference time
static double g_last_inference_time = 0.0;

/**
 * Helper to convert jstring to std::string
 */
static std::string jstringToString(JNIEnv* env, jstring jstr) {
    if (jstr == nullptr) return "";
    
    const char* chars = env->GetStringUTFChars(jstr, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(jstr, chars);
    
    return result;
}

/**
 * Helper to convert std::string to jstring
 */
static jstring stringToJstring(JNIEnv* env, const std::string& str) {
    return env->NewStringUTF(str.c_str());
}

/**
 * Helper to throw Java exception
 */
static void throwJavaException(JNIEnv* env, const char* className, const char* message) {
    jclass exClass = env->FindClass(className);
    if (exClass != nullptr) {
        env->ThrowNew(exClass, message);
    }
}

// =============================================================================
// Native method implementations
// =============================================================================

JNIEXPORT jlong JNICALL Java_com_selfdriving_engine_SelfDrivingEngine_nativeCreate(
    JNIEnv* env,
    jobject obj,
    jstring modelPath,
    jboolean useGpu,
    jint gpuDeviceId
) {
    try {
        InferenceConfig config;
        config.model_path = jstringToString(env, modelPath);
        config.use_gpu = useGpu;
        config.gpu_device_id = gpuDeviceId;
        
        // Create engine
        InferenceEngine* engine = new InferenceEngine(config);
        
        std::cout << "[JNI] Created inference engine at " << engine << std::endl;
        
        return reinterpret_cast<jlong>(engine);
        
    } catch (const std::exception& e) {
        throwJavaException(env, "java/lang/RuntimeException", e.what());
        return 0;
    }
}

JNIEXPORT void JNICALL Java_com_selfdriving_engine_SelfDrivingEngine_nativeDestroy(
    JNIEnv* env,
    jobject obj,
    jlong enginePtr
) {
    if (enginePtr != 0) {
        InferenceEngine* engine = reinterpret_cast<InferenceEngine*>(enginePtr);
        std::cout << "[JNI] Destroying inference engine at " << engine << std::endl;
        delete engine;
    }
}

JNIEXPORT jboolean JNICALL Java_com_selfdriving_engine_SelfDrivingEngine_nativeInitialize(
    JNIEnv* env,
    jobject obj,
    jlong enginePtr
) {
    if (enginePtr == 0) {
        throwJavaException(env, "java/lang/NullPointerException", "Engine pointer is null");
        return JNI_FALSE;
    }
    
    try {
        InferenceEngine* engine = reinterpret_cast<InferenceEngine*>(enginePtr);
        bool success = engine->initialize();
        return success ? JNI_TRUE : JNI_FALSE;
        
    } catch (const std::exception& e) {
        throwJavaException(env, "java/lang/RuntimeException", e.what());
        return JNI_FALSE;
    }
}

JNIEXPORT jboolean JNICALL Java_com_selfdriving_engine_SelfDrivingEngine_nativeIsReady(
    JNIEnv* env,
    jobject obj,
    jlong enginePtr
) {
    if (enginePtr == 0) return JNI_FALSE;
    
    InferenceEngine* engine = reinterpret_cast<InferenceEngine*>(enginePtr);
    return engine->isReady() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jfloatArray JNICALL Java_com_selfdriving_engine_SelfDrivingEngine_nativeInfer(
    JNIEnv* env,
    jobject obj,
    jlong enginePtr,
    jfloatArray state
) {
    if (enginePtr == 0) {
        throwJavaException(env, "java/lang/NullPointerException", "Engine pointer is null");
        return nullptr;
    }
    
    try {
        InferenceEngine* engine = reinterpret_cast<InferenceEngine*>(enginePtr);
        
        // Get input data
        jsize inputLen = env->GetArrayLength(state);
        jfloat* inputData = env->GetFloatArrayElements(state, nullptr);
        
        std::vector<float> inputVec(inputData, inputData + inputLen);
        
        env->ReleaseFloatArrayElements(state, inputData, JNI_ABORT);
        
        // Run inference
        InferenceResult result = engine->infer(inputVec);
        
        if (!result.success) {
            throwJavaException(env, "java/lang/RuntimeException", 
                             result.error_message.c_str());
            return nullptr;
        }
        
        // Store inference time
        g_last_inference_time = result.inference_time_ms;
        
        // Create output array
        jfloatArray output = env->NewFloatArray(result.actions.size());
        env->SetFloatArrayRegion(output, 0, result.actions.size(), 
                                 result.actions.data());
        
        return output;
        
    } catch (const std::exception& e) {
        throwJavaException(env, "java/lang/RuntimeException", e.what());
        return nullptr;
    }
}

JNIEXPORT jfloatArray JNICALL Java_com_selfdriving_engine_SelfDrivingEngine_nativeInferBatch(
    JNIEnv* env,
    jobject obj,
    jlong enginePtr,
    jfloatArray states,
    jint batchSize
) {
    if (enginePtr == 0) {
        throwJavaException(env, "java/lang/NullPointerException", "Engine pointer is null");
        return nullptr;
    }
    
    try {
        InferenceEngine* engine = reinterpret_cast<InferenceEngine*>(enginePtr);
        
        // Get input data
        jsize inputLen = env->GetArrayLength(states);
        jfloat* inputData = env->GetFloatArrayElements(states, nullptr);
        
        std::vector<float> inputVec(inputData, inputData + inputLen);
        
        env->ReleaseFloatArrayElements(states, inputData, JNI_ABORT);
        
        // Run batch inference
        InferenceResult result = engine->inferBatch(inputVec, batchSize);
        
        if (!result.success) {
            throwJavaException(env, "java/lang/RuntimeException", 
                             result.error_message.c_str());
            return nullptr;
        }
        
        g_last_inference_time = result.inference_time_ms;
        
        // Create output array
        jfloatArray output = env->NewFloatArray(result.actions.size());
        env->SetFloatArrayRegion(output, 0, result.actions.size(), 
                                 result.actions.data());
        
        return output;
        
    } catch (const std::exception& e) {
        throwJavaException(env, "java/lang/RuntimeException", e.what());
        return nullptr;
    }
}

JNIEXPORT jfloatArray JNICALL Java_com_selfdriving_engine_SelfDrivingEngine_nativeInferFromImage(
    JNIEnv* env,
    jobject obj,
    jlong enginePtr,
    jbyteArray imageData,
    jint height,
    jint width
) {
    if (enginePtr == 0) {
        throwJavaException(env, "java/lang/NullPointerException", "Engine pointer is null");
        return nullptr;
    }
    
    try {
        InferenceEngine* engine = reinterpret_cast<InferenceEngine*>(enginePtr);
        
        // Get image data
        jsize imageLen = env->GetArrayLength(imageData);
        jbyte* imageBytes = env->GetByteArrayElements(imageData, nullptr);
        
        // Run inference
        InferenceResult result = engine->inferFromImage(
            reinterpret_cast<unsigned char*>(imageBytes),
            height,
            width
        );
        
        env->ReleaseByteArrayElements(imageData, imageBytes, JNI_ABORT);
        
        if (!result.success) {
            throwJavaException(env, "java/lang/RuntimeException", 
                             result.error_message.c_str());
            return nullptr;
        }
        
        g_last_inference_time = result.inference_time_ms;
        
        // Create output array
        jfloatArray output = env->NewFloatArray(result.actions.size());
        env->SetFloatArrayRegion(output, 0, result.actions.size(), 
                                 result.actions.data());
        
        return output;
        
    } catch (const std::exception& e) {
        throwJavaException(env, "java/lang/RuntimeException", e.what());
        return nullptr;
    }
}

JNIEXPORT jstring JNICALL Java_com_selfdriving_engine_SelfDrivingEngine_nativeGetExecutionProvider(
    JNIEnv* env,
    jobject obj,
    jlong enginePtr
) {
    if (enginePtr == 0) {
        return stringToJstring(env, "UNKNOWN");
    }
    
    InferenceEngine* engine = reinterpret_cast<InferenceEngine*>(enginePtr);
    return stringToJstring(env, engine->getExecutionProvider());
}

JNIEXPORT jdouble JNICALL Java_com_selfdriving_engine_SelfDrivingEngine_nativeGetLastInferenceTime(
    JNIEnv* env,
    jobject obj,
    jlong enginePtr
) {
    return g_last_inference_time;
}

JNIEXPORT void JNICALL Java_com_selfdriving_engine_SelfDrivingEngine_nativeWarmup(
    JNIEnv* env,
    jobject obj,
    jlong enginePtr,
    jint iterations
) {
    if (enginePtr == 0) return;
    
    InferenceEngine* engine = reinterpret_cast<InferenceEngine*>(enginePtr);
    engine->warmup(iterations);
}

JNIEXPORT jdouble JNICALL Java_com_selfdriving_engine_SelfDrivingEngine_nativeBenchmark(
    JNIEnv* env,
    jobject obj,
    jlong enginePtr,
    jint iterations
) {
    if (enginePtr == 0) return -1.0;
    
    InferenceEngine* engine = reinterpret_cast<InferenceEngine*>(enginePtr);
    return engine->benchmark(iterations);
}
