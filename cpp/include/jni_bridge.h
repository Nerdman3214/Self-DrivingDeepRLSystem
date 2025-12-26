#ifndef SELF_DRIVING_JNI_BRIDGE_H
#define SELF_DRIVING_JNI_BRIDGE_H

/**
 * JNI Bridge Header
 * 
 * Defines the native methods that will be called from Java.
 * Generated method signatures must match the Java native declarations.
 */

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// SelfDrivingEngine native methods
// =============================================================================

/**
 * Create a new inference engine instance
 * 
 * @param env JNI environment
 * @param obj Java object reference
 * @param modelPath Path to ONNX model file
 * @param useGpu Whether to use GPU acceleration
 * @param gpuDeviceId GPU device ID (if using GPU)
 * @return Native pointer to engine instance, or 0 on failure
 */
JNIEXPORT jlong JNICALL Java_com_selfdriving_engine_SelfDrivingEngine_nativeCreate(
    JNIEnv* env,
    jobject obj,
    jstring modelPath,
    jboolean useGpu,
    jint gpuDeviceId
);

/**
 * Destroy the inference engine instance
 * 
 * @param env JNI environment
 * @param obj Java object reference
 * @param enginePtr Native pointer to engine
 */
JNIEXPORT void JNICALL Java_com_selfdriving_engine_SelfDrivingEngine_nativeDestroy(
    JNIEnv* env,
    jobject obj,
    jlong enginePtr
);

/**
 * Initialize the engine and load the model
 * 
 * @param env JNI environment
 * @param obj Java object reference
 * @param enginePtr Native pointer to engine
 * @return true if initialization successful
 */
JNIEXPORT jboolean JNICALL Java_com_selfdriving_engine_SelfDrivingEngine_nativeInitialize(
    JNIEnv* env,
    jobject obj,
    jlong enginePtr
);

/**
 * Check if engine is ready for inference
 * 
 * @param env JNI environment
 * @param obj Java object reference
 * @param enginePtr Native pointer to engine
 * @return true if ready
 */
JNIEXPORT jboolean JNICALL Java_com_selfdriving_engine_SelfDrivingEngine_nativeIsReady(
    JNIEnv* env,
    jobject obj,
    jlong enginePtr
);

/**
 * Run inference on a state observation
 * 
 * @param env JNI environment
 * @param obj Java object reference
 * @param enginePtr Native pointer to engine
 * @param state Flattened observation array (normalized to [0,1])
 * @return Action array [steering, gas, brake]
 */
JNIEXPORT jfloatArray JNICALL Java_com_selfdriving_engine_SelfDrivingEngine_nativeInfer(
    JNIEnv* env,
    jobject obj,
    jlong enginePtr,
    jfloatArray state
);

/**
 * Run batch inference
 * 
 * @param env JNI environment
 * @param obj Java object reference
 * @param enginePtr Native pointer to engine
 * @param states Batch of flattened observations
 * @param batchSize Number of observations in batch
 * @return Batched action array
 */
JNIEXPORT jfloatArray JNICALL Java_com_selfdriving_engine_SelfDrivingEngine_nativeInferBatch(
    JNIEnv* env,
    jobject obj,
    jlong enginePtr,
    jfloatArray states,
    jint batchSize
);

/**
 * Run inference from image data
 * 
 * @param env JNI environment
 * @param obj Java object reference
 * @param enginePtr Native pointer to engine
 * @param imageData RGB image bytes
 * @param height Image height
 * @param width Image width
 * @return Action array
 */
JNIEXPORT jfloatArray JNICALL Java_com_selfdriving_engine_SelfDrivingEngine_nativeInferFromImage(
    JNIEnv* env,
    jobject obj,
    jlong enginePtr,
    jbyteArray imageData,
    jint height,
    jint width
);

/**
 * Get execution provider name
 * 
 * @param env JNI environment
 * @param obj Java object reference
 * @param enginePtr Native pointer to engine
 * @return Provider name (e.g., "CUDA", "CPU")
 */
JNIEXPORT jstring JNICALL Java_com_selfdriving_engine_SelfDrivingEngine_nativeGetExecutionProvider(
    JNIEnv* env,
    jobject obj,
    jlong enginePtr
);

/**
 * Get last inference time in milliseconds
 * 
 * @param env JNI environment
 * @param obj Java object reference
 * @param enginePtr Native pointer to engine
 * @return Inference time in ms
 */
JNIEXPORT jdouble JNICALL Java_com_selfdriving_engine_SelfDrivingEngine_nativeGetLastInferenceTime(
    JNIEnv* env,
    jobject obj,
    jlong enginePtr
);

/**
 * Warm up the engine
 * 
 * @param env JNI environment
 * @param obj Java object reference
 * @param enginePtr Native pointer to engine
 * @param iterations Number of warmup iterations
 */
JNIEXPORT void JNICALL Java_com_selfdriving_engine_SelfDrivingEngine_nativeWarmup(
    JNIEnv* env,
    jobject obj,
    jlong enginePtr,
    jint iterations
);

/**
 * Run benchmark
 * 
 * @param env JNI environment
 * @param obj Java object reference
 * @param enginePtr Native pointer to engine
 * @param iterations Number of benchmark iterations
 * @return Average inference time in ms
 */
JNIEXPORT jdouble JNICALL Java_com_selfdriving_engine_SelfDrivingEngine_nativeBenchmark(
    JNIEnv* env,
    jobject obj,
    jlong enginePtr,
    jint iterations
);

#ifdef __cplusplus
}
#endif

#endif // SELF_DRIVING_JNI_BRIDGE_H
