#define NOMINMAX
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <numbers>
#include <iostream>
#include <cmath>
#include <Windows.h>
#include "memory.hpp"
#include "server_dll.hpp"
#include "offsets.hpp"
#include "VisCheck.h"
#include <deque>

struct Vector3d {
    float x, y, z;

    Vector3d operator+(const Vector3d& v) const {
        return { x + v.x, y + v.y, z + v.z };
    }

    Vector3d operator-(const Vector3d& v) const {
        return { x - v.x, y - v.y, z - v.z };
    }

    Vector3d operator*(float s) const {
        return { x * s, y * s, z * s };
    }

    Vector3d operator/(float f) const {
        return { x / f, y / f, z / f };
    }

    float length() const {
        return std::sqrt(x * x + y * y + z * z);

    }
    float distance(const Vector3d& other) const {
        return (*this - other).length();
    }
    Vector3d normalized() const {
        float l = length(); return l > 0 ? *this / l : Vector3d{ 0,0,0 };
    }

};

struct Vector2 {
    float x, y;
};

struct QAngle {
    float pitch;
    float yaw;
    float roll;
};

struct Quaternion {
    float x;
    float y;
    float z;
    float w;
};

struct ShortName {
    char name[10];

    bool operator==(const ShortName& v) const {
        for (int i = 0; i < 10; ++i) {
            if (name[i] != v.name[i])
                return false;
        }
        return true;
    }
};

struct CBoneData {
    Vector3d pos;
    float scale;
    Quaternion rot;
};

struct Bones {
    CBoneData bones[23];
};

struct Matrix4x4 {
    float m[16];
};

struct PlayerState {
    Bones bones;
    Vector3d eyePosition;
    Vector3d originPosition;
    Vector3d absVelocity;
    QAngle viewAngles;
    uint64_t buttonMask;
    uint32_t health;
    uint32_t ammoInClip;
    uint16_t weaponID;
    bool isReloading;
    uint64_t steamId;
    ShortName name;
    bool isAlive;
    bool isImmuneToDamage;
    float duckAmount;
    bool onGround;
    float fallVelocity;
    bool isWalking;
    uint8_t teamNum;
    uint32_t tick;
};

struct PlayerAction {
    // Continuous: Aiming
    float deltaPitch; // Vertical aim movement
    float deltaYaw; // Horizontal aim movement

    // Discrete: Movement Axes (-1, 0, or 1)
    float moveX;      // S = -1, None or both = 0, W = 1
    float moveY;      // A = -1, None or both = 0, D = 1

    // Binary: Combat and Utility
    bool isShooting;   // Left click
    bool isJumping;    // Space
    bool isCrouching;  // Ctrl
    bool isReloading;  // R
    bool isWalking;    // Shift
};

struct EnemyObservation {
    bool isValid; // stays true for 90 ticks after an enemy is seen, then becomes false until a new enemy is seen
    bool isVisible; // not visible if not rendered on screen
    bool isImmune; // is immune to damage in deathmatch mode
    float proximityScore;
    float targetDeltaPitch;
    float targetDeltaYaw;
    float memoryConfidence; // [0.0f, 1.0f] (1.0 - ticks/90.0)
    ShortName name; // just for calculations, not included in the training data.
    Vector3d lastSeenPos; // just for calculations, not included in the training data.
    int ticksSinceLastSeen; // just for calculations, not included in the training data.
};

struct DamageObservation {
    bool isValid;            // 1.0 if hit recently, 0.0 if empty padding
    float damageDeltaPitch;    // Relative horizontal angle (-1.0 to 1.0)
    float damageDeltaYaw;  // Relative vertical angle (-1.0 to 1.0)
    float damageIntensity;   // timer that starts at 1.0 when hit and decays to 0.0 over 30 ticks
    Vector3d source; //field to prepare the data, to be deleted after preparation
    ShortName name; //field to prepare the data, to be deleted after preparation
};

struct SelfObservation {
    // GLOBAL: Where am I? (Lets the bot memorize the map)
    float globalPosX;
    float globalPosY;
    float globalPosZ;

    // Where am I looking at?
    float pitch;
    float yaw;

    // Movement State
    float velocityForward;  // Relative to crosshair (W/S)
    float velocityRight;    // Relative to crosshair (A/D)
    float velocityUp;       // Falling or jumping speed

    // Stance & Environment Flags
    float isGrounded;
    float duckAmount;       // 0.0 (standing) to 1.0 (fully crouched)

    // Combat State
    float healthPercentage; // 0.0 to 1.0
    float ammoPercentage;   // 0.0 to 1.0
    float isReloading; // 0.0 or 1.0
};

struct FlatFrame {
    float values[65]; // 9+7*5+4*2+13
};

struct Hitbox {
    Vector3d minBounds;
    Vector3d maxBounds;
    float radius;
    //uint8_t boneId;
};

struct HitboxArray {
    Hitbox hitboxes[19];
};

struct InferenceInput {
    float values[316];
};

class BehavioralCloningModel {
private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session;
    Ort::MemoryInfo memory_info;

public:
    // Constructor: Loads the model ONCE
    BehavioralCloningModel(const wchar_t* model_path)
        : env(ORT_LOGGING_LEVEL_WARNING, "BehavioralCloning"),
        session(env, model_path, session_options),
        memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU))
    {
        // Optional: Optimize threading for intra-op operations
        session_options.SetIntraOpNumThreads(2);
    }

    // The inference function you asked about
    void runInference(const InferenceInput& inputStruct, float* outActions) {
        // Define shapes: Batch size 1, 316 input features; Batch size 1, 9 output actions
        std::vector<int64_t> input_shape = { 1, 316 };
        std::vector<int64_t> output_shape = { 1, 9 };

        // Create input tensor directly pointing to struct's memory (zero-copy)
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(inputStruct.values), // Point to the raw array
            316,                                    // Fixed element count
            input_shape.data(),
            input_shape.size()
        );

        const char* input_names[] = { "state_input" };
        const char* output_names[] = { "action_output" };

        // Run the model
        auto output_tensors = session.Run(
            Ort::RunOptions{ nullptr },
            input_names, &input_tensor, 1,
            output_names, 1
        );

        // Extract raw pointer from output tensor and copy to outActions array
        float* float_arr = output_tensors.front().GetTensorMutableData<float>();
        for (int i = 0; i < 9; ++i) {
            outActions[i] = float_arr[i];
        }
    }
};

// Helper function to convert binary action logits (indices 4-8) into true/false states
inline bool logitToBool(float logit) {
    // Sigmoid function: 1 / (1 + e^(-x))
    float prob = 1.0f / (1.0f + std::exp(-logit));
    return prob > 0.5f;
}
// Function to linearly normalize a value to the range [-1.0, 1.0]
float normalize(float value, float origMin, float origMax) {
    float normalized = -1.0f + ((value - origMin) * 2.0f) / (origMax - origMin);
    float clamped = std::clamp(normalized, -1.f, 1.f);
    return clamped;
}

// Function to normalize angle. 35 for mouse movement, 60 for the angle between crosshair and enemy
float normalizeAngle(float deltaAngle, float maxAngle = 35) {
    float power = 0.5; // e.g., 0.5 for square-root scaling

    float clamped = std::clamp(deltaAngle, -maxAngle, maxAngle);
    float sign = (clamped >= 0.0f) ? 1.0f : -1.0f;

    // Ratio in [0.0, 1.0]
    float ratio = std::abs(clamped) / maxAngle;

    // Apply power curve and re-apply sign
    return sign * std::pow(ratio, power);

    return 0.f;
}

float denormalizeAngle(float normalizedValue, float maxAngle = 35) {
    float power = 0.5; // Must match the power used in normalization

    float sign = (normalizedValue >= 0.0f) ? 1.0f : -1.0f;
    float absVal = std::abs(normalizedValue);

    // Reverse the power curve: raise to the power of (1.0 / power)
    float ratio = std::pow(absVal, 1.0f / power);

    // Scale back up to maxAngle and re-apply the sign
    return 2* sign * ratio * maxAngle; // TODO REMOVE 2
}


// Proximity Score[0.0 = Far / Irrelevant, 1.0 = Right on top of player]
// combat_radius: Distance in units where enemy importance drops by ~63% (3000.0f is max distance for dust2)
float getProximityScore(float distance, float combatRadius = 500.0f) {
    return std::exp(-distance / combatRadius);
}

Vector3d getLocalVelocity(Vector3d absVelocity, float yaw) {
    float yawRad = yaw * (std::numbers::pi_v<float> / 180.0f);

    float cosYaw = std::cos(yawRad);
    float sinYaw = std::sin(yawRad);

    // Apply 2D rotation matrix by -yaw:
    // localX = absX * cos(yaw) + absY * sin(yaw)   (Forward velocity)
    // localY = -absX * sin(yaw) + absY * cos(yaw)  (Strafe/Side velocity)
    Vector3d relativeVelocity;
    relativeVelocity.x = absVelocity.x * cosYaw + absVelocity.y * sinYaw;
    relativeVelocity.y = -absVelocity.x * sinYaw + absVelocity.y * cosYaw;
    relativeVelocity.z = absVelocity.z; // Vertical velocity stays the same

    return relativeVelocity;
}

Vector3d Cross(const Vector3d& a, const Vector3d& b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

Vector3d RotateVectorByQuaternion(const Vector3d& v, const Quaternion& q)
{
    Vector3d qv = { q.x, q.y, q.z };
    Vector3d t = Cross(qv, v) * 2.0f;
    return v + (t * q.w) + Cross(qv, t);
}

bool WorldToScreen(const Vector3d& pos, Vector2& screen, float m[16], int width, int height)
{
    float clipX = pos.x * m[0] + pos.y * m[1] + pos.z * m[2] + m[3];
    float clipY = pos.x * m[4] + pos.y * m[5] + pos.z * m[6] + m[7];
    float clipW = pos.x * m[12] + pos.y * m[13] + pos.z * m[14] + m[15];

    if (clipW < 0.1f)
        return false;

    float ndcX = clipX / clipW;
    float ndcY = clipY / clipW;

    if (ndcX < -1.0f || ndcX > 1.0f || ndcY < -1.0f || ndcY > 1.0f)
        return false;

    screen.x = (width / 2.0f) * (ndcX + 1.0f);
    screen.y = (height / 2.0f) * (1.0f - ndcY);

    //std::cout << "on the point: " << screen.x << "  " << screen.y << std::endl;

    return true;
}

HitboxArray getHitboxArray(const Bones& bones) {
    HitboxArray hitboxes;

    // Helper macro/lambda can clean this up, but keeping it direct for clarity:
    auto transformHitbox = [&](int index, int boneIdx, const Vector3d& localMin, const Vector3d& localMax, float radius) {
        const auto& bone = bones.bones[boneIdx];
        hitboxes.hitboxes[index] = {
            bone.pos + RotateVectorByQuaternion(localMin, bone.rot),
            bone.pos + RotateVectorByQuaternion(localMax, bone.rot),
            radius
        };
        };

    // 0 : 7 head
    transformHitbox(0, 7, { -1, 1.8, 0 }, { 3.5, 0.2, 0 }, 4.3f);
    // 1 : 6 neck
    transformHitbox(1, 6, { 0, -0.4, 0 }, { 1.4, -0.2, 0 }, 3.5f);
    // 2 : 1 hip
    transformHitbox(2, 1, { -2.7, 1.1, -3.2 }, { -2.7, 1.1, 3.2 }, 6.0f);
    // 3 : 2 spine
    transformHitbox(3, 2, { 1.4, 0.8, 3.1 }, { 1.4, 0.8, -3.1 }, 6.0f);
    // 4 : 3 spine
    transformHitbox(4, 3, { 3.8, 0.8, -2.4 }, { 3.8, 0.4, 2.4 }, 6.5f);
    // 5 : 4 spine
    transformHitbox(5, 4, { 4.8, 0.15, -4.1 }, { 4.8, 0.15, 4.1 }, 6.2f);
    // 6 : 5 spine
    transformHitbox(6, 5, { 2.5, -0.6, -6 }, { 2.5, -0.6, 6 }, 5.0f);
    // 7 : 17 upper leg L
    transformHitbox(7, 17, { 1.3, -0.2, 0 }, { 16.5, -0.7, 0 }, 5.0f);
    // 8 : 20 upper leg R
    transformHitbox(8, 20, { -1.3, 0, -0.6 }, { -16.5, 0, -0.7 }, 5.0f);
    // 9 : 18 lower leg L
    transformHitbox(9, 18, { 0.1, -0.4, 0.2 }, { 17, -0.4, 0.7 }, 4.0f);
    // 10 : 21 lower leg R
    transformHitbox(10, 21, { -0.1, 0, -0.2 }, { -17, 0.4, -0.7 }, 4.0f);
    // 11 : 19 foot L
    transformHitbox(11, 19, { -0, -3.43, -0.52 }, { 8, 0.74, 0.33 }, 2.6f);
    // 12 : 22 foot R
    transformHitbox(12, 22, { -7.98, -0.75, -0.27 }, { -0.02, 3.44, 0.58 }, 2.6f);
    // 13 : 11 hand L
    transformHitbox(13, 11, { 0, 0.3, 0 }, { 3.59, 1.15, 0.11 }, 2.3f);
    // 14 : 15 hand R
    transformHitbox(14, 15, { 0, -0.3, 0.02 }, { -3.44, -1.17, -0.09 }, 2.3f);
    // 15 : 9 lower arm L
    transformHitbox(15, 9, { 0, 0, 0 }, { 11.2, 0, 0 }, 3.3f);
    // 16 : 10 upper arm L
    transformHitbox(16, 10, { 0, 0, 0 }, { 10, 0, 0 }, 3.0f);
    // 17 : 13 lower arm R
    transformHitbox(17, 13, { 0, 0, 0 }, { -11.2, 0, 0 }, 3.3f);
    // 18 : 14 upper arm R
    transformHitbox(18, 14, { 0, 0, 0 }, { -10, 0, -0.5 }, 3.0f);

    return hitboxes;
}

void getCapsulePeripheralPoints(Hitbox hitbox, Vector3d eyePos, std::vector<Vector3d>& outPoints, int arcPoints = 7, int linePoints = 4) {
    outPoints.clear();
    if (arcPoints < 2) arcPoints = 2;
    if (linePoints < 2) linePoints = 2;

    // Total points will be exactly (arcPoints * 2) + (linePoints * 2) without duplicates
    outPoints.reserve((arcPoints * 2) + (linePoints * 2));

    Vector3d p1 = hitbox.minBounds; // Bottom sphere center
    Vector3d p2 = hitbox.maxBounds; // Top sphere center
    double r = hitbox.radius;

    Vector3d capDir = p2 - p1;
    double axisLen = capDir.length();
    if (axisLen < 0.0001) return;
    capDir = capDir.normalized();

    // Perspective Horizon Frame for both Spheres
    auto getHorizonFrame = [&](const Vector3d& sphereCenter, Vector3d& outCenter, double& outRadius, Vector3d& outForward) {
        Vector3d toEye = eyePos - sphereCenter;
        double d = toEye.length();
        outForward = toEye.normalized();
        if (d > r) {
            double h = (r * r) / d;
            outRadius = sqrt(r * r - h * h);
            outCenter = sphereCenter + outForward * h;
        }
        else {
            outRadius = r;
            outCenter = sphereCenter;
        }
        };

    Vector3d c2, c1, f2, f1;
    double r2, r1;
    getHorizonFrame(p2, c2, r2, f2);
    getHorizonFrame(p1, c1, r1, f1);

    // Calculate a uniform 3D Side Direction
    Vector3d midCenter = (c1 + c2) * 0.5;
    Vector3d toEyeMid = (eyePos - midCenter).normalized();
    Vector3d sideDir = Cross(capDir, toEyeMid).normalized();

    // Singularity protection (Looking straight down the capsule axis)
    double midDot = (toEyeMid.x * capDir.x) + (toEyeMid.y * capDir.y) + (toEyeMid.z * capDir.z);
    if (midDot > 0.999 || midDot < -0.999) {
        sideDir = (abs(capDir.y) > 0.99) ? Vector3d{ 1, 0, 0 } : Cross(Vector3d{ 0, 1, 0 }, capDir).normalized();
    }

    // Local "Up" vectors for the circles that point AWAY from the capsule body
    Vector3d up2 = Cross(f2, sideDir).normalized();
    double up2Dot = (up2.x * capDir.x) + (up2.y * capDir.y) + (up2.z * capDir.z);
    if (up2Dot < 0) up2 = up2 * -1.0;

    Vector3d up1 = Cross(f1, sideDir).normalized();
    double up1Dot = (up1.x * capDir.x) + (up1.y * capDir.y) + (up1.z * capDir.z);
    if (up1Dot > 0) up1 = up1 * -1.0;

    // --- Top Outer Arc ---
    // Exclude the very last point (Left Tangent), because the Left Line starts there.
    for (int i = 0; i < arcPoints; ++i) {
        double t = i / static_cast<double>(arcPoints - 1);
        double theta = t * std::numbers::pi_v<double>;
        Vector3d offset = (sideDir * cos(theta)) + (up2 * sin(theta));
        outPoints.push_back(c2 + offset * r2);
    }
    outPoints.pop_back(); // Remove duplicate at the end of the arc

    // --- Left Connecting Line ---
    // Exclude the very last point (Bottom Left Tangent), because the Bottom Arc starts there.
    Vector3d topLeftTangent = c2 - sideDir * r2;
    Vector3d bottomLeftTangent = c1 - sideDir * r1;
    for (int i = 0; i < linePoints; ++i) {
        double t = i / static_cast<double>(linePoints - 1);
        outPoints.push_back(topLeftTangent * (1.0 - t) + bottomLeftTangent * t);
    }
    outPoints.pop_back(); // Remove duplicate at the end of the line

    // --- Bottom Outer Arc ---
    // Exclude the very last point (Bottom Right Tangent), because the Right Line starts there.
    for (int i = 0; i < arcPoints; ++i) {
        double t = i / static_cast<double>(arcPoints - 1);
        double theta = t * std::numbers::pi_v<double>;
        Vector3d offset = ((sideDir * -1.0) * cos(theta)) + (up1 * sin(theta));
        outPoints.push_back(c1 + offset * r1);
    }
    outPoints.pop_back(); // Remove duplicate at the end of the arc

    // --- Right Connecting Line ---
    // Exclude the very last point (Top Right Tangent), because it matches the exact first point of the Top Arc!
    Vector3d bottomRightTangent = c1 + sideDir * r1;
    Vector3d topRightTangent = c2 + sideDir * r2;
    for (int i = 0; i < linePoints; ++i) {
        double t = i / static_cast<double>(linePoints - 1);
        outPoints.push_back(bottomRightTangent * (1.0 - t) + topRightTangent * t);
    }
    outPoints.pop_back(); // Remove duplicate at the end of the line (loop closes perfectly)
}

Matrix4x4 CalculateUnrealViewProjection(
    const Vector3d& camPos,
    float pitchDeg, float yawDeg,
    float fovHorizontalDeg = 106.260205f, // Extracted from memory dump
    float screenWidth = 1920.0f,
    float screenHeight = 1080.0f
) {
    // Get the base camera axes
    float pitch = -pitchDeg * (3.14159265359f / 180.0f);
    float yaw = yawDeg * (3.14159265359f / 180.0f);

    float CP = std::cos(pitch); float SP = std::sin(pitch);
    float CY = std::cos(yaw);   float SY = std::sin(yaw);

    // calculated Right vector is opposite, we negate the Yaw component
    Vector3d F = { CP * CY,   CP * SY,   SP };
    Vector3d R = { SY,       -CY,        0.0f }; // Negated components
    Vector3d U = { -SP * CY,  -SP * SY,   CP };

    // Apply the translation
    float tx = -(R.x * camPos.x + R.y * camPos.y + R.z * camPos.z);
    float ty = -(U.x * camPos.x + U.y * camPos.y + U.z * camPos.z);
    float tz = -(F.x * camPos.x + F.y * camPos.y + F.z * camPos.z);

    // Calculate the Lens zoom scalars
    float halfFovRad = (fovHorizontalDeg * 0.5f) * (3.14159265359f / 180.0f);
    float gx = 1.0f / std::tan(halfFovRad);
    float aspect = screenWidth / screenHeight;
    float gy = gx * aspect;

    // Standard gaming Infinite Far-Plane depth scaling
    float nearPlane = 6.5f;
    float farPlane = 102522.0f; // 100000.0f  1.0002522f
    float Q = farPlane / (farPlane - nearPlane);

    Matrix4x4 vp;

    // Row 0: Right axis scaled by Horizontal Zoom
    vp.m[0] = R.x * gx;  vp.m[1] = R.y * gx;  vp.m[2] = R.z * gx;  vp.m[3] = tx * gx;

    // Row 1: Up axis scaled by Vertical Zoom
    vp.m[4] = U.x * gy;  vp.m[5] = U.y * gy;  vp.m[6] = U.z * gy;  vp.m[7] = ty * gy;

    // Row 2: Z-Buffer mapping row (Forward axis * Q)
    vp.m[8] = F.x * Q;   vp.m[9] = F.y * Q;   vp.m[10] = F.z * Q;   vp.m[11] = (tz * Q) - (nearPlane * Q);

    // Row 3: The "Perspective Divide" trigger row (Pure Forward axis)
    vp.m[12] = F.x;       vp.m[13] = F.y;       vp.m[14] = F.z;       vp.m[15] = tz;

    return vp;
}

Vector2 calculateAngle(const Vector3d& source, const QAngle& viewAngles, const Vector3d& destination) {
    Vector3d delta = destination - source;
    float hypotenuse = std::sqrt(delta.x * delta.x + delta.y * delta.y);

    float targetPitch = std::atan2(-delta.z, hypotenuse) * (180.0f / 3.14159265f);
    float targetYaw = std::atan2(delta.y, delta.x) * (180.0f / 3.14159265f);

    Vector2 relativeAngles;
    relativeAngles.x = targetPitch - viewAngles.pitch;
    relativeAngles.y = targetYaw - viewAngles.yaw;

    while (relativeAngles.y > 180.0f)  relativeAngles.y -= 360.0f;
    while (relativeAngles.y < -180.0f) relativeAngles.y += 360.0f;

    if (relativeAngles.x > 160.0f)  relativeAngles.x = 160.0f;
    if (relativeAngles.x < -160.0f) relativeAngles.x = -160.0f;

    return relativeAngles;
}

bool get_target_point_front(Vector3d& target, const Bones& bones, const Vector3d& eyePosition, const QAngle& viewAngles, const VisCheck& checker) {
    target = { 0 };
    int visiblePoints = 0;
    HitboxArray hitboxes = getHitboxArray(bones);
    int priorityArray[19] = { 0, 1, 6, 5, 4, 3, 2, 15, 17, 16, 18, 13, 14, 7, 8, 9, 10, 11, 12 };

    Matrix4x4 viewMatrix = CalculateUnrealViewProjection(eyePosition, viewAngles.pitch, viewAngles.yaw);

    for (int i = 0; i < 19; ++i) {
        Hitbox hitbox = hitboxes.hitboxes[priorityArray[i]];
        std::vector<Vector3d> points;
        getCapsulePeripheralPoints(hitbox, eyePosition, points, 7, 4);
        for (const Vector3d& point : points) {
            Vector2 screen;
            if (WorldToScreen(point, screen, viewMatrix.m, 1920, 1080)) {
                if (checker.IsPointVisible({ eyePosition.x, eyePosition.y, eyePosition.z }, { point.x, point.y, point.z })) {
                    visiblePoints++;
                    target = target + point;
                }
            }
        }
        if (visiblePoints > 0) {
            target = target / static_cast<float>(visiblePoints);
            return true;
        }
    }

    return false;
}

bool inNameList(const ShortName& name, const std::vector<ShortName>& playersList) {
    return std::find(playersList.begin(), playersList.end(), name) != playersList.end();
}

void analyzeDamage(const std::vector<PlayerState>& curStates, const std::vector<PlayerState>& prevStates, const VisCheck& checker, std::vector<DamageObservation>& damageObservations)
{
    int currentHealth = 0;
    int previousHealth = 0;
    PlayerState targetState;
    bool found = false;
    for (const auto& curState : curStates) {
        if (curState.steamId != 0) {
            currentHealth = curState.health;
            targetState = curState;
            found = true;
            break;
        }
    }
    if (!found) {
        return;
    }
    found = false;
    for (const auto& prevState : prevStates) {
        if (prevState.steamId != 0) {
            previousHealth = prevState.health;
            found = true;
            break;
        }
    }

    if (!found) {
        // the player wasnt in the previous Tick. must be spawned this tick.
        // TODO copy curState to prevState
    }

    int framesToFade = 30;
    float fadeAmount = 1.0f / framesToFade;
    // lower the damage observation intensity by 1/30, if its 0 make other fields 0.
    for (auto& damageObservation : damageObservations) {
        damageObservation.damageIntensity -= fadeAmount;
        if (damageObservation.damageIntensity <= 0) {
            damageObservation = { 0 };
        }
        else {
            Vector2 angle = calculateAngle(targetState.eyePosition, targetState.viewAngles, damageObservation.source);
            damageObservation.damageDeltaPitch = normalizeAngle(angle.x, 180);
            damageObservation.damageDeltaYaw = normalizeAngle(angle.y, 180);
        }
    }

    if (currentHealth < previousHealth) {
        //std::cout << nameList.name[whichPlayer].name << " Got damaged at tick " << curStates[0].tick << " from " << previousHealth << " to " << currentHealth << std::endl;
    }
    else {
        return;
    }

    for (const auto& curState : curStates) {
        if (curState.steamId != 0) {
            continue;
        }
        for (const auto& prevState : prevStates) {
            if (curState.name == prevState.name) {
                int currentAmmo = curState.ammoInClip;
                int previousAmmo = prevState.ammoInClip;
                if (currentAmmo < previousAmmo) {
                    Vector3d target;
                    bool visible = get_target_point_front(target, targetState.bones, curState.eyePosition, curState.viewAngles, checker);
                    if (visible) {
                        //std::cout << " damage " << curState.eyePosition.x << "|" << curState.eyePosition.y << "|" << curState.eyePosition.z << " || "
                        //    << targetState.eyePosition.x << "|" << targetState.eyePosition.y << "|" << targetState.eyePosition.z << std::endl;
                        //std::cout << " angle " << curState.viewAngles.pitch << "|" << curState.viewAngles.yaw << std::endl;


                        //Vector2 angle = calculateAngle(curState.eyePosition, curState.viewAngles, targetState.eyePosition);
                        Vector2 angle = calculateAngle(targetState.eyePosition, targetState.viewAngles, curState.eyePosition);
                        //std::cout << "Player " << nameList.name[whichPlayer].name << " damaged by " << curState.name.name << " at tick " << curState.tick
                        //    << " angle " << angle.x << " | " << angle.y << std::endl;

                        //std::cout << "shooter pos " << curState.eyePosition.x << " | " << curState.eyePosition.y << " looking at "
                        //    << curState.viewAngles.pitch << "|" << curState.viewAngles.yaw << std::endl;
                        //std::cout << "target pos " << targetState.eyePosition.x << " | " << targetState.eyePosition.y << " looking at "
                        //    << targetState.viewAngles.pitch << "|" << targetState.viewAngles.yaw << std::endl;
                        //if the name is in damage observations override it, otherwise override the observation with the least intensity
                        DamageObservation* leastIntenseDO = &damageObservations.front();
                        for (auto& damageObservation : damageObservations) {
                            if (damageObservation.name == curState.name) {
                                damageObservation.isValid = true;
                                damageObservation.damageDeltaPitch = normalizeAngle(angle.x, 180);
                                damageObservation.damageDeltaYaw = normalizeAngle(angle.y, 180);
                                damageObservation.damageIntensity = 1.0;
                                damageObservation.source = curState.eyePosition;
                                damageObservation.name = curState.name;
                                return; // TODO why is this here?
                            }
                            if (damageObservation.damageIntensity < leastIntenseDO->damageIntensity) {
                                leastIntenseDO = &damageObservation;
                            }
                        }
                        leastIntenseDO->isValid = true;
                        leastIntenseDO->damageDeltaPitch = angle.x;
                        leastIntenseDO->damageDeltaYaw = angle.y;
                        leastIntenseDO->damageIntensity = 1.0;
                        leastIntenseDO->source = curState.eyePosition;
                        leastIntenseDO->name = curState.name;
                    }
                }
            }
        }
    }

}

void replaceEnemyObservation(std::vector<EnemyObservation>& enemyObservations, const EnemyObservation& newObservation) {

    // Override observation with the same name if it exists
    for (auto& enemyObservation : enemyObservations) {
        if (enemyObservation.name == newObservation.name) {
            enemyObservation = newObservation;
            return;
        }
    }

    // Override not valid observation if there is any
    for (auto& enemyObservation : enemyObservations) {
        if (!enemyObservation.isValid) {
            enemyObservation = newObservation;
            return;
        }
    }

    // Override oldest not visible if there is any
    EnemyObservation* observationToReplace = nullptr;
    int maxAge = -1;

    for (auto& enemyObservation : enemyObservations) {
        if (enemyObservation.isValid && !enemyObservation.isVisible) {
            int age = enemyObservation.ticksSinceLastSeen;
            if (age > maxAge) {
                maxAge = age;
                observationToReplace = &enemyObservation;
            }
        }
    }
    if (observationToReplace != nullptr) {
        *observationToReplace = newObservation;
        return;
    }


    // Override valid, visible with highest delta angle
    observationToReplace = nullptr;
    float maxAngularDistance = -1.f;

    for (auto& enemyObservation : enemyObservations) {
        if (enemyObservation.isValid && enemyObservation.isVisible) {
            float AngularDistance = std::hypot(enemyObservation.targetDeltaPitch, enemyObservation.targetDeltaYaw);
            if (AngularDistance > maxAngularDistance) {
                maxAngularDistance = AngularDistance;
                observationToReplace = &enemyObservation;
            }
        }
    }
    if (observationToReplace != nullptr) {
        *observationToReplace = newObservation;
        return;
    }

    std::cout << "never should have happened :)" << std::endl; // should never happen
}

void reorderEnemyObservations(std::vector<EnemyObservation>& enemyObservations) {
    std::sort(enemyObservations.begin(), enemyObservations.end(), [](const EnemyObservation& a, const EnemyObservation& b) {
        // --- Handle validity grouping ---
        // Valid elements come before invalid elements
        if (a.isValid != b.isValid) {
            return a.isValid > b.isValid; // true (1) comes before false (0)
        }

        // If both are invalid, their relative order doesn't matter
        if (!a.isValid) {
            return false;
        }

        // --- Handle visibility grouping (both are valid here) ---
        // Visible elements come before non-visible elements
        if (a.isVisible != b.isVisible) {
            return a.isVisible > b.isVisible;
        }

        // --- Tie-breakers within the same group ---
        if (a.isVisible) {
            // 1st Priority: Order by least hypot(deltaPitch, deltaYaw)
            // Using squared values avoids slow sqrt() calls
            float distSqA = (a.targetDeltaPitch * a.targetDeltaPitch) + (a.targetDeltaYaw * a.targetDeltaYaw);
            float distSqB = (b.targetDeltaPitch * b.targetDeltaPitch) + (b.targetDeltaYaw * b.targetDeltaYaw);
            return distSqA < distSqB;
        }
        else {
            // 2nd Priority: Order by age / ticksSinceLastSeen (ascending/youngest first)
            return a.ticksSinceLastSeen < b.ticksSinceLastSeen;
        }
        });
}

void analyzeEnemy(const std::vector<PlayerState>& curStates, const std::vector<PlayerState>& prevStates, const VisCheck& checker, std::vector<EnemyObservation>& enemyObservations)
{
    // TODO should this function return isPlayerAlive? so that we wont produce processed state data for our ml model?

    PlayerState playerState;
    //NameList alivePlayersList;
    std::vector<ShortName> alivePlayersList;
    std::vector<ShortName> seenPlayersList;
    for (const auto& curState : curStates) {
        if (curState.steamId != 0) {
            playerState = curState;
        }
        else {
            //if (!inNameList(curState.name, alivePlayersList)) { //TODO investigate the changes
            alivePlayersList.push_back(curState.name);
            //}
        }
    }

    std::vector<EnemyObservation> newObservations;
    for (const auto& curState : curStates) {
        if (curState.steamId != 0) {
            continue;
        }
        else {
            Vector3d target;
            bool visible = get_target_point_front(target, curState.bones, playerState.eyePosition, playerState.viewAngles, checker);
            if (visible) {
                seenPlayersList.push_back(curState.name);
                Vector2 angle = calculateAngle(playerState.eyePosition, playerState.viewAngles, target);

                float distance = playerState.eyePosition.distance(target);
                float proximity = getProximityScore(distance);
                EnemyObservation newObservation = {
                    .isValid = true,
                    .isVisible = true,
                    .isImmune = curState.isImmuneToDamage,
                    .proximityScore = proximity,
                    .targetDeltaPitch = normalizeAngle(angle.x ,60), // max angle is around 60 in the game with default settings
                    .targetDeltaYaw = normalizeAngle(angle.y ,60), // max angle is around 60 in the game with default settings
                    .memoryConfidence = 1,
                    .name = curState.name,
                    .lastSeenPos = target,
                    .ticksSinceLastSeen = 0
                };

                newObservations.push_back(newObservation);

                //replaceEnemyObservation(enemyObservations, newObservation);

            }
        }
    }

    int ticksToForget = 90;
    for (auto& enemyObservation : enemyObservations) {
        if (enemyObservation.isValid && enemyObservation.isVisible) {
            // if enemy was valid and visible but not visible this tick make it not visible
            if (!inNameList(enemyObservation.name, seenPlayersList)) {
                enemyObservation.isVisible = false;
            }
        }
        if (enemyObservation.isValid && !enemyObservation.isVisible) {
            enemyObservation.ticksSinceLastSeen += 1;
            enemyObservation.memoryConfidence = 1.0 - (enemyObservation.ticksSinceLastSeen / 90.0);
            // update delta angle, proximityScore
            Vector2 angle = calculateAngle(playerState.eyePosition, playerState.viewAngles, enemyObservation.lastSeenPos);
            enemyObservation.targetDeltaPitch = normalizeAngle(angle.x, 60); // max angle is around 60 in the game with default settings
            enemyObservation.targetDeltaYaw = normalizeAngle(angle.y, 60); // max angle is around 60 in the game with default settings
            float distance = playerState.eyePosition.distance(enemyObservation.lastSeenPos);
            enemyObservation.proximityScore = getProximityScore(distance);
        }
        if (enemyObservation.ticksSinceLastSeen > ticksToForget) {
            enemyObservation = { 0 };
        }
        if (!inNameList(enemyObservation.name, alivePlayersList)) { // forget enemy if dies this tick
            enemyObservation = { 0 };
        }
        if (!enemyObservation.isValid) {
            enemyObservation = { 0 };
        }
        //enemyObservation.isImmune = ; // TODO figure out how to update this, now only updates when enemy is visible, which might actually be ok
    }

    // If new observations are enough to fill enemyObservations fill them
    if (newObservations.size() >= enemyObservations.size()) {
        reorderEnemyObservations(newObservations);

        // Store the target size before changing anything
        size_t targetSize = enemyObservations.size();

        // Safely replace the contents with the range from newObservations
        enemyObservations.assign(newObservations.begin(), newObservations.begin() + targetSize);
        return;
    }

    // there is less newObservations than enemyObservations slots
    for (auto& newObservation : newObservations) {
        replaceEnemyObservation(enemyObservations, newObservation);
    }

    reorderEnemyObservations(enemyObservations);
}

void analyzeSelf(const std::vector<PlayerState>& curStates, const std::vector<PlayerState>& prevStates, SelfObservation& selfObs) {
    bool found = false;
    PlayerState curState;
    PlayerState prevState;
    for (const auto& state : curStates) {
        // TODO, also get the prev state(if born this tick and prev state doesnt exist, copy current) to calculate delta view angles
        if (state.steamId != 0) {
            curState = state;
            found = true;
        }
    }

    if (!found) { // if not present in current tick(not alive)
        selfObs = { 0 };
        return;
    }

    prevState = curState;
    for (const auto& state : prevStates) {
        // TODO, also get the prev state(if born this tick and prev state doesnt exist, copy current) to calculate delta view angles
        if (state.steamId != 0) {
            prevState = state;
        }
    }

    // TODO normalize all values to -1.0 to 1.0
    // Dust 2
    // x: 1788 -2204
    // y: 3118 -1164
    // z: 252 -131
    selfObs.globalPosX = normalize(curState.originPosition.x, -2204.f, 1788.f);
    selfObs.globalPosY = normalize(curState.originPosition.y, -1164.f, 3118.f);
    selfObs.globalPosZ = normalize(curState.originPosition.z, -131.f, 252.f);
    selfObs.pitch = normalize(curState.viewAngles.pitch, -180.f, 180.f);
    selfObs.yaw = normalize(curState.viewAngles.yaw, -180.f, 180.f);

    Vector3d localVel = getLocalVelocity(curState.absVelocity, curState.viewAngles.yaw);
    selfObs.velocityForward = normalize(localVel.x, -250.f, 250.f);
    selfObs.velocityRight = normalize(localVel.y, -250.f, 250.f);
    selfObs.velocityUp = normalize(localVel.z, -300.f, 300.f);

    //std::cout << "fallVelocity " << curState.fallVelocity << " |velocityUp " << curState.absVelocity.z << std::endl;

    selfObs.isGrounded = curState.onGround;
    selfObs.duckAmount = curState.duckAmount;
    selfObs.healthPercentage = static_cast<float>(curState.health) / 100.0f; // 0.0 to 1.0
    selfObs.ammoPercentage = static_cast<float>(curState.ammoInClip) / 30.0f;   // 0.0 to 1.0
    selfObs.isReloading = curState.isReloading;

}

FlatFrame getFlatFrame(SelfObservation selfObs, std::vector<EnemyObservation> enemyObs, std::vector<DamageObservation> damageObs, PlayerAction action) {

    // Check if there is exactly 5 enemyObs and 2 damageObs
    if (enemyObs.size() != 5 || damageObs.size() != 2) {
        FlatFrame emptyFrame;
        std::memset(emptyFrame.values, 0, sizeof(emptyFrame.values));
        return emptyFrame;
    }

    float arr[65];

    // SelfObservation
    arr[0] = selfObs.globalPosX;
    arr[1] = selfObs.globalPosY;
    arr[2] = selfObs.globalPosZ;
    arr[3] = selfObs.pitch;
    arr[4] = selfObs.yaw;
    arr[5] = selfObs.velocityForward;
    arr[6] = selfObs.velocityRight;
    arr[7] = selfObs.velocityUp;
    arr[8] = selfObs.isGrounded;
    arr[9] = selfObs.duckAmount;
    arr[10] = selfObs.healthPercentage;
    arr[11] = selfObs.ammoPercentage;
    arr[12] = selfObs.isReloading;

    // EnemyObservation 0
    arr[13] = enemyObs[0].isValid ? 1.f : 0.f;
    arr[14] = enemyObs[0].isVisible ? 1.f : 0.f;
    arr[15] = enemyObs[0].isImmune ? 1.f : 0.f;
    arr[16] = enemyObs[0].proximityScore;
    arr[17] = enemyObs[0].targetDeltaPitch;
    arr[18] = enemyObs[0].targetDeltaYaw;
    arr[19] = enemyObs[0].memoryConfidence;

    // EnemyObservation 1
    arr[20] = enemyObs[1].isValid ? 1.f : 0.f;
    arr[21] = enemyObs[1].isVisible ? 1.f : 0.f;
    arr[22] = enemyObs[1].isImmune ? 1.f : 0.f;
    arr[23] = enemyObs[1].proximityScore;
    arr[24] = enemyObs[1].targetDeltaPitch;
    arr[25] = enemyObs[1].targetDeltaYaw;
    arr[26] = enemyObs[1].memoryConfidence;

    // EnemyObservation 2
    arr[27] = enemyObs[2].isValid ? 1.f : 0.f;
    arr[28] = enemyObs[2].isVisible ? 1.f : 0.f;
    arr[29] = enemyObs[2].isImmune ? 1.f : 0.f;
    arr[30] = enemyObs[2].proximityScore;
    arr[31] = enemyObs[2].targetDeltaPitch;
    arr[32] = enemyObs[2].targetDeltaYaw;
    arr[33] = enemyObs[2].memoryConfidence;

    // EnemyObservation 3
    arr[34] = enemyObs[3].isValid ? 1.f : 0.f;
    arr[35] = enemyObs[3].isVisible ? 1.f : 0.f;
    arr[36] = enemyObs[3].isImmune ? 1.f : 0.f;
    arr[37] = enemyObs[3].proximityScore;
    arr[38] = enemyObs[3].targetDeltaPitch;
    arr[39] = enemyObs[3].targetDeltaYaw;
    arr[40] = enemyObs[3].memoryConfidence;

    // EnemyObservation 4
    arr[41] = enemyObs[4].isValid ? 1.f : 0.f;
    arr[42] = enemyObs[4].isVisible ? 1.f : 0.f;
    arr[43] = enemyObs[4].isImmune ? 1.f : 0.f;
    arr[44] = enemyObs[4].proximityScore;
    arr[45] = enemyObs[4].targetDeltaPitch;
    arr[46] = enemyObs[4].targetDeltaYaw;
    arr[47] = enemyObs[4].memoryConfidence;

    // DamageObservation 0
    arr[48] = damageObs[0].isValid ? 1.f : 0.f;
    arr[49] = damageObs[0].damageDeltaPitch;
    arr[50] = damageObs[0].damageDeltaYaw;
    arr[51] = damageObs[0].damageIntensity;

    // DamageObservation 1
    arr[52] = damageObs[1].isValid ? 1.f : 0.f;
    arr[53] = damageObs[1].damageDeltaPitch;
    arr[54] = damageObs[1].damageDeltaYaw;
    arr[55] = damageObs[1].damageIntensity;

    // PlayerAction
    arr[56] = action.deltaPitch;
    arr[57] = action.deltaYaw;
    arr[58] = action.moveX;
    arr[59] = action.moveY;
    arr[60] = action.isShooting ? 1.f : 0.f;
    arr[61] = action.isJumping ? 1.f : 0.f;
    arr[62] = action.isCrouching ? 1.f : 0.f;
    arr[63] = action.isReloading ? 1.f : 0.f;
    arr[64] = action.isWalking ? 1.f : 0.f;


    // Package into the struct and return
    FlatFrame result;
    std::memcpy(result.values, arr, sizeof(arr));
    return result;
}

InferenceInput constructInferenceInput(const std::deque<FlatFrame>& historyBuffer, const FlatFrame& currentFrame) {
    InferenceInput input;
    int targetIndex = 0;

    // Copy specific frames from the history buffer
    // Indices corresponding to: historyBuffer[15], historyBuffer[11], historyBuffer[7], historyBuffer[3]
    int frameIndices[] = { 15, 11, 7, 3 };

    for (int frameIdx : frameIndices) {
        // Ensure the history buffer has enough elements to prevent out-of-bounds errors
        if (historyBuffer.size() > frameIdx) {
            const float* src = historyBuffer[historyBuffer.size() - 1 - frameIdx].values;
            // Assuming you want the full 65 floats from each of these frames
            std::memcpy(&input.values[targetIndex], src, sizeof(float) * 65);
        }
        else {
            // Handle padding or zero-filling if history is too short
            std::memset(&input.values[targetIndex], 0, sizeof(float) * 65);
        }
        targetIndex += 65; // 4 frames * 65 = 260 floats total so far
    }

    // Copy the last 9 floats from the current frame
    // A FlatFrame has 65 floats, so the last 9 start at index (65 - 9 = 56)
    int sourceStartIndex = 65 - 9; // 56
    for (int i = 0; i < 9; ++i) {
        input.values[targetIndex++] = currentFrame.values[sourceStartIndex + i];
    }

    while (targetIndex < 316) {
        input.values[targetIndex++] = 0.0f;
    }

    return input;
}

void printActions(float actions[]) {
    // --- PARSE ACTIONS ---
// Continuous Actions (Direct values)
    float deltaPitch = denormalizeAngle(actions[0]);
    float deltaYaw = denormalizeAngle(actions[1]);
    float moveX = actions[2]; // Strafe (-1 to 1)
    float moveY = actions[3]; // Move forward/back (-1 to 1)

    // Binary Actions (Need sigmoid thresholding)
    float threshold = 0.2f;

    // Map to directional boolean flags
    bool pressW = (moveY > threshold);   // Move Forward
    bool pressS = (moveY < -threshold);  // Move Backward
    bool pressA = (moveX < -threshold);  // Move Left (Strafe Left)
    bool pressD = (moveX > threshold);   // Move Right (Strafe Right)
    bool isShooting = logitToBool(actions[4]);
    bool isJumping = logitToBool(actions[5]);
    bool isCrouching = logitToBool(actions[6]);
    bool isReloading = logitToBool(actions[7]);
    bool isWalking = logitToBool(actions[8]);

    // Debug output to verify
    std::cout << "Aim Delta Pitch: " << deltaPitch << " | moveX " << moveX << " | moveY " << moveY
        << " | Shooting: " << (isShooting ? "YES" : "NO") << std::endl;
}

PlayerAction getPlayerAction(float actions[]) {
    PlayerAction action = {
        .deltaPitch = actions[0],
        .deltaYaw = actions[1],
        .moveX = actions[2],
        .moveY = actions[3],
        .isShooting = logitToBool(actions[4]),
        .isJumping = logitToBool(actions[5]),
        .isCrouching = logitToBool(actions[6]),
        .isReloading = logitToBool(actions[7]),
        .isWalking = logitToBool(actions[8]),
    };

    return action;
}

void applyActions(float actions[], const HANDLE driver, const std::uintptr_t client) {
    // --- PARSE ACTIONS ---
    // Continuous Actions (Direct values)
    float deltaPitch = denormalizeAngle(actions[0]);
    float deltaYaw = denormalizeAngle(actions[1]);
    float moveX = actions[2]; // Strafe (-1 to 1)
    float moveY = actions[3]; // Move forward/back (-1 to 1)

    //Vector3d targetViewAngles = { 0 };
    QAngle targetViewAngles = driver::read_memory<QAngle>(driver, client + cs2_dumper::offsets::client_dll::dwViewAngles);
    targetViewAngles.pitch += deltaPitch;
    targetViewAngles.yaw += deltaYaw;
    driver::write_memory(driver, client + cs2_dumper::offsets::client_dll::dwViewAngles, targetViewAngles); // TODO PLAY AROUND WITH THIS

    // Binary Actions (Need sigmoid thresholding)
    float threshold = 0.2f;

    // Map to directional boolean flags
    bool pressW = (moveY > threshold);   // Move Forward
    bool pressS = (moveY < -threshold);  // Move Backward
    bool pressA = (moveX < -threshold);  // Move Left (Strafe Left)
    bool pressD = (moveX > threshold);   // Move Right (Strafe Right)
    bool isShooting = logitToBool(actions[4]);
    bool isJumping = logitToBool(actions[5]);
    bool isCrouching = logitToBool(actions[6]);
    bool isReloading = logitToBool(actions[7]);
    bool isWalking = logitToBool(actions[8]);

    // Debug output to verify
    std::cout << "Aim Delta Pitch: " << deltaPitch << " | moveX " << moveX << " | moveY " << moveY
        << " | Shooting: " << (isShooting ? "YES" : "NO") << std::endl;
}

void printEnemyObservations(const std::vector<EnemyObservation>& enemyObservations) {

    std::cout << "==============================================================" << std::endl;
    for (auto& enemyObservation : enemyObservations) {
        std::cout << enemyObservation.isValid << "|" <<
            enemyObservation.isVisible << "|proximity " <<
            enemyObservation.proximityScore << "|" <<
            enemyObservation.targetDeltaPitch << "|" <<
            enemyObservation.targetDeltaYaw << "|" <<
            enemyObservation.name.name << "|" <<
            enemyObservation.lastSeenPos.x << "|" << enemyObservation.lastSeenPos.y << "|" << enemyObservation.lastSeenPos.z << "|" <<
            enemyObservation.ticksSinceLastSeen << std::endl;
        std::cout << "--------------------------------------------------------------" << std::endl;
    }
}

int main() {
    const DWORD pid = get_process_id(L"cs2.exe");
    if (pid == 0) {
        std::cout << "Failed to find cs2.\n";
        std::cin.get();
        return 1;
    }
    const HANDLE driver = CreateFileW(L"\\\\.\\xdriver", GENERIC_READ, 0, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL, nullptr);
    if (driver == INVALID_HANDLE_VALUE) {
        std::cout << "Failed to create driver handle.\n";
        std::cin.get();
        return 1;
    }
    if (driver::attach_to_process(driver, pid) == true) {
        std::cout << "Attachment successful.\n";
    }
    else {
        std::cout << "Failed to attach to process.\n";
        std::cin.get();
        return 1;
    }
    
    const std::uintptr_t server = get_module_base(pid, L"server.dll");

    if (server != 0) {
        std::cout << "Getting server.dll module_base was successful.\n";
    }
    else {
        std::cout << "Failed to get server.dll module_base.\n";
        std::cin.get();
        return 1;
    }

    const std::uintptr_t client = get_module_base(pid, L"client.dll");

    if (client != 0) {
        std::cout << "Getting client.dll module_base was successful.\n";
    }
    else {
        std::cout << "Failed to get client.dll module_base.\n";
        std::cin.get();
        return 1;
    }

    // Load model once at startup
    BehavioralCloningModel model(L"behavioral_cloning_model.onnx");
    const VisCheck checker = VisCheck::VisCheck("de_dust2.tri");

    // Simulate game state buffer
    // Must contain exactly 316 floats representing t-16 through t0
    std::vector<float> current_state_buffer(316, 0.0f);

    // Array to hold the 9 output actions predicted by the model
    float actions[9];
    std::vector<DamageObservation> damageObservations;
    damageObservations.push_back({ 0 });
    damageObservations.push_back({ 0 });

    std::vector<EnemyObservation> enemyObservations;
    enemyObservations.push_back({ 0 });
    enemyObservations.push_back({ 0 });
    enemyObservations.push_back({ 0 });
    enemyObservations.push_back({ 0 });
    enemyObservations.push_back({ 0 });

    SelfObservation selfObservation = { 0 };

    std::deque<FlatFrame> historyBuffer;

    // --- GAME LOOP SIMULATION (Runs every tick/frame) ---
    bool loopEnabled = false;
    bool messagePrinted = false;
    uint32_t tickBase = 0;
    while (true) {
        if (GetAsyncKeyState(VK_END))
        {
            // Force the window to close so draw::render() returns
            //PostMessage(FindWindowA("OverlayWindowClass", "Overlay Window"), WM_CLOSE, 0, 0); // TODO uncomment for drawing boxes
            break;
        }
        const int triggerKey = 0x71; // F2 key
        if (GetAsyncKeyState(triggerKey) & 1) {
            loopEnabled = !loopEnabled;
            messagePrinted = false;
        }
        if (loopEnabled && !messagePrinted) {
            std::cout << "Bot is ON" << std::endl;
            messagePrinted = true;
        }
        else if (!loopEnabled && !messagePrinted) {
            std::cout << "Bot is OFF" << std::endl;
            messagePrinted = true;
        }

        std::vector<PlayerState> prevStates;
        std::vector<PlayerState> curStates;
        if (loopEnabled) {
            // TODO do everything here
            uintptr_t entityList = driver::read_memory<uintptr_t>(driver, server + cs2_dumper::offsets::server_dll::dwEntityList);

            if (tickBase == 0) {
                uintptr_t globalVarsPtr = driver::read_memory<uintptr_t>(driver, server + cs2_dumper::offsets::server_dll::dwGlobalVars);
                tickBase = driver::read_memory<uint32_t>(driver, globalVarsPtr + 0x44);
                if (tickBase == 0)
                    continue;
            }
            else {
                uintptr_t globalVarsPtr = driver::read_memory<uintptr_t>(driver, server + cs2_dumper::offsets::server_dll::dwGlobalVars);
                uint32_t currentTickBase = driver::read_memory<uint32_t>(driver, globalVarsPtr + 0x44);
                if (currentTickBase == tickBase) {
                    continue;
                }
                if (currentTickBase > tickBase + 1) {
                    std::cout << "JUMP HAPPENED!!!!!!!!!!!!!!" << tickBase << std::endl;
                }
                tickBase = currentTickBase;
            }
            std::cout << "tick: " << tickBase << std::endl;


            for (int i = 0; i < 64; i++)
            {
                uintptr_t listEntry = driver::read_memory<uintptr_t>(driver, entityList + (0x8 * (i & 0x7FFF) >> 9) + 0x10);

                if (!listEntry)
                    continue;

                uintptr_t controller = driver::read_memory<uintptr_t>(driver, listEntry + 0x70 * (i & 0x1FF));

                if (!controller)
                    continue;

                uint32_t pawnHandle = driver::read_memory<uint32_t>(driver, controller + cs2_dumper::schemas::server_dll::CBasePlayerController::m_hPawn);
                uintptr_t pawnListEntry = driver::read_memory<uintptr_t>(driver, entityList + (0x8 * ((i & 0x7FFF) >> 9)) + 0x10);
                uintptr_t pawn = driver::read_memory<uintptr_t>(driver, pawnListEntry + 0x70 * (pawnHandle & 0x1FF));

                ShortName name = driver::read_memory<ShortName>(driver, controller + cs2_dumper::schemas::server_dll::CBasePlayerController::m_iszPlayerName);

                if (name.name[0] == '\0')
                    continue;

                bool isAlive = driver::read_memory<bool>(driver, controller + cs2_dumper::schemas::server_dll::CCSPlayerController::m_bPawnIsAlive);
                if (!isAlive)
                    continue;

                uintptr_t weaponServices = driver::read_memory<uintptr_t>(driver, pawn + cs2_dumper::schemas::server_dll::CBasePlayerPawn::m_pWeaponServices);
                uint32_t weaponHandle = driver::read_memory<uint32_t>(driver, weaponServices + cs2_dumper::schemas::server_dll::CPlayer_WeaponServices::m_hActiveWeapon);
                uintptr_t weaponListEntry = driver::read_memory<uintptr_t>(driver, entityList + (0x8 * ((weaponHandle & 0x7FFF) >> 9)) + 0x10);
                if (!weaponListEntry)
                    continue;

                uintptr_t weapon = driver::read_memory<uintptr_t>(driver, weaponListEntry + 0x70 * (weaponHandle & 0x1FF));
                if (!weapon)
                    continue;

                uint16_t weaponID = driver::read_memory<uint16_t>(driver, weapon + cs2_dumper::schemas::server_dll::CEconEntity::m_AttributeManager + cs2_dumper::schemas::server_dll::CAttributeContainer::m_Item + cs2_dumper::schemas::server_dll::CEconItemView::m_iItemDefinitionIndex);
                uint32_t ammoInClip = driver::read_memory<uint32_t>(driver, weapon + cs2_dumper::schemas::server_dll::CBasePlayerWeapon::m_iClip1);
                bool inReload = driver::read_memory<bool>(driver, weapon + cs2_dumper::schemas::server_dll::CCSWeaponBase::m_bInReload);

                auto flags = driver::read_memory<std::uint32_t>(driver, pawn + cs2_dumper::schemas::server_dll::CBaseEntity::m_fFlags);
                bool onGround = flags & (1 << 0);

                uintptr_t bodyComponent = driver::read_memory<uintptr_t>(driver, pawn + cs2_dumper::schemas::server_dll::CBaseEntity::m_CBodyComponent);
                uintptr_t gameScene = driver::read_memory<uintptr_t>(driver, bodyComponent + cs2_dumper::schemas::server_dll::CBodyComponent::m_pSceneNode);
                Vector3d origin = driver::read_memory<Vector3d>(driver, gameScene + cs2_dumper::schemas::server_dll::CGameSceneNode::m_vecAbsOrigin);
                Vector3d viewOffset = driver::read_memory<Vector3d>(driver, pawn + cs2_dumper::schemas::server_dll::CBaseModelEntity::m_vecViewOffset);
                Vector3d eyePosition = origin + viewOffset;
                Vector3d vecAbsVelocity = driver::read_memory<Vector3d>(driver, pawn + cs2_dumper::schemas::server_dll::CBaseEntity::m_vecAbsVelocity);
                QAngle viewAngles = driver::read_memory<QAngle>(driver, pawn + cs2_dumper::schemas::server_dll::CCSPlayerPawn::m_angEyeAngles);

                bool isWalking = driver::read_memory<bool>(driver, pawn + cs2_dumper::schemas::server_dll::CCSPlayerPawn::m_bIsWalking);
                uint8_t teamNum = driver::read_memory<uint8_t>(driver, pawn + cs2_dumper::schemas::server_dll::CBaseEntity::m_iTeamNum);
                uintptr_t CSkeletonInstance = driver::read_memory<uintptr_t>(driver, gameScene + cs2_dumper::schemas::server_dll::CSkeletonInstance::m_modelState + cs2_dumper::schemas::server_dll::CBodyComponentSkeletonInstance::m_skeletonInstance);
                Bones bones = driver::read_memory<Bones>(driver, CSkeletonInstance);
                uintptr_t movementServices = driver::read_memory<uintptr_t>(driver, pawn + cs2_dumper::schemas::server_dll::CBasePlayerPawn::m_pMovementServices);
                float duckAmount = driver::read_memory<float>(driver, movementServices + cs2_dumper::schemas::server_dll::CCSPlayer_MovementServices::m_flDuckAmount);
                uint64_t steamId = driver::read_memory<uint64_t>(driver, controller + cs2_dumper::schemas::server_dll::CBasePlayerController::m_steamID);
                bool isImmune = driver::read_memory<bool>(driver, pawn + cs2_dumper::schemas::server_dll::CCSPlayerPawn::m_bGunGameImmunity);
                uint32_t health = driver::read_memory<uint32_t>(driver, pawn + cs2_dumper::schemas::server_dll::CBaseEntity::m_iHealth);
                float fallVelocity = driver::read_memory<float>(driver, movementServices + cs2_dumper::schemas::server_dll::CPlayer_MovementServices_Humanoid::m_flFallVelocity);

                PlayerState playerState;
                playerState.bones = std::move(bones);
                playerState.eyePosition = std::move(eyePosition);
                playerState.originPosition = std::move(origin);
                playerState.absVelocity = std::move(vecAbsVelocity);
                playerState.viewAngles = std::move(viewAngles);
                playerState.buttonMask = 0;
                playerState.health = std::move(health);
                playerState.ammoInClip = std::move(ammoInClip);
                playerState.weaponID = std::move(weaponID);
                playerState.isReloading = std::move(inReload);
                playerState.steamId = std::move(steamId);
                playerState.name = std::move(name);
                playerState.isAlive = std::move(isAlive);
                playerState.isImmuneToDamage = std::move(isImmune);
                playerState.duckAmount = std::move(duckAmount);
                playerState.onGround = std::move(onGround);
                playerState.fallVelocity = std::move(fallVelocity);
                playerState.isWalking = std::move(isWalking);
                playerState.teamNum = std::move(teamNum);
                playerState.tick = std::move(tickBase);

                curStates.push_back(playerState);
            }

            // TODO ANALYZE
            analyzeDamage(curStates, prevStates, checker, damageObservations);
            analyzeEnemy(curStates, prevStates, checker, enemyObservations);
            analyzeSelf(curStates, prevStates, selfObservation);
            FlatFrame currentFrame = getFlatFrame(selfObservation, enemyObservations, damageObservations, PlayerAction{ 0 });

            printEnemyObservations(enemyObservations);

            if (prevStates.size() == 0 || historyBuffer.size() == 0) {
                historyBuffer.clear();
                for (int i = 0; i < 16; i++) {
                    historyBuffer.push_front(currentFrame);
                }
            }

            InferenceInput input  = constructInferenceInput(historyBuffer, currentFrame);
            model.runInference(input, actions);

            //printActions(actions);
            applyActions(actions, driver, client);


            // TODO apply action
            currentFrame = getFlatFrame(selfObservation, enemyObservations, damageObservations, getPlayerAction(actions));
            // put the current frame to history buffer and pop the oldest one
            historyBuffer.push_front(currentFrame);
            historyBuffer.pop_back();
            //

            prevStates = std::move(curStates);
            curStates.clear();
        }
    }
    

    return 0;
}