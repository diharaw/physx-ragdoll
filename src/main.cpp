#define NOMINMAX
#include <application.h>
#include <mesh.h>
#include <camera.h>
#include <material.h>
#include <memory>
#include <iostream>
#include <stack>
#include "skeletal_mesh.h"
#include "anim_sample.h"
#include "anim_local_to_global.h"
#include "anim_offset.h"
#include "anim_ragdoll.h"

using namespace physx;

//#define RIFLE_MESH

// Uniform buffer data structure.
struct ObjectUniforms
{
    DW_ALIGNED(16)
    glm::mat4 model;
};

struct GlobalUniforms
{
    DW_ALIGNED(16)
    glm::mat4 view;
    DW_ALIGNED(16)
    glm::mat4 projection;
};

#define CAMERA_FAR_PLANE 10000.0f

const std::string kJointNames[] = {
    "Pelvis",
    "Middle Spine",
    "Neck",
    "Head",
    "Left Upper Leg",
    "Left Knee",
    "Left Ankle",
    "Left Foot",
    "Right Upper Leg",
    "Right Knee",
    "Right Angle",
    "Right Foot",
    "Left Upper Arm",
    "Left Elbow",
    "Left Lower Arm",
    "Left Hand",
    "Right Upper Arm",
    "Right Elbow",
    "Right Lower Arm",
    "Right Hand"
};

struct RagdollDesc
{
    std::string pelvis;
    std::string mid_spine;
    std::string neck;
    std::string head;
    std::string left_upper_leg;
    std::string left_knee;
    std::string left_ankle;
    std::string left_foot;
    std::string right_upper_leg;
    std::string right_knee;
    std::string right_ankle;
    std::string right_foot;
    std::string left_upper_arm;
    std::string left_elbow;
    std::string left_lower_arm;
    std::string left_hand;
    std::string right_upper_arm;
    std::string right_elbow;
    std::string right_lower_arm;
    std::string right_hand;
};

class AnimationStateMachine : public dw::Application
{
protected:
    // -----------------------------------------------------------------------------------------------------------------------------------

    bool init(int argc, const char* argv[]) override
    {
        m_selected_joints.resize(20);

        for (int i = 0; i < 20; i++)
            m_selected_joints[i] = 0;

        initialize_physics();

        // Create GPU resources.
        if (!create_shaders())
            return false;

        if (!create_uniform_buffer())
            return false;

        // Load mesh.
        if (!load_mesh())
            return false;

        m_joint_names.push_back(" -");

        for (int i = 0; i < m_skeletal_mesh->skeleton()->num_bones(); i++)
            m_joint_names.push_back(m_skeletal_mesh->skeleton()->joints()[i].name);

        DW_LOG_INFO("Loaded Mesh!");

        // Load animations.
        if (!load_animations())
            return false;

        DW_LOG_INFO("Loaded Animations!");

        // Create camera.
        create_camera();

        for (int i = 0; i < MAX_BONES; i++)
            m_pose_transforms.transforms[i] = glm::mat4(1.0f);

        m_index_stack.reserve(256);
        m_joint_pos.reserve(256);

        DW_LOG_INFO("Loading Done!");

        update_camera();

        m_joint_mat.resize(m_skeletal_mesh->skeleton()->num_bones());

        create_ragdoll();

        m_scene->simulate(1.0f / 60.0f);
        m_scene->fetchResults(true);

        m_ragdoll.set_kinematic(true);

        return true;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update(double delta) override
    {
        // Debug GUI
        gui();

        // Update camera.
        update_camera();

        // Update global uniforms.
        Joint* joints = m_skeletal_mesh->skeleton()->joints();

        for (int i = 0; i < m_skeletal_mesh->skeleton()->num_bones(); i++)
            m_pose_transforms.transforms[i] = glm::inverse(joints[i].inverse_bind_pose);

        update_global_uniforms(m_global_uniforms);
        update_object_uniforms(m_character_transforms);

        // Bind and set viewport.
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, m_width, m_height);

        // Clear default framebuffer.
        glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Bind states.
        glEnable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);

        // Update Skeleton
        update_animations();

        if (!m_pause_sim)
        {
            m_scene->simulate(1.0f / 60.0f);
            m_scene->fetchResults(true);
        }

        // Render Mesh.
        if (m_visualize_mesh)
            render_skeletal_meshes();

        // Render Joints.
        if (m_visualize_joints)
            visualize_skeleton(m_skeletal_mesh->skeleton());

        // Render Bones.
        if (m_visualize_bones)
            visualize_bones(m_skeletal_mesh->skeleton());

        if (m_physx_debug_draw)
            render_physx_debug();

        // Render debug draw.
        m_debug_draw.render(nullptr, m_width, m_height, m_debug_mode ? m_debug_camera->m_view_projection : m_main_camera->m_view_projection);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void shutdown() override
    {
        m_scene->release();
        m_dispatcher->release();
        m_physics->release();
        m_foundation->release();
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void window_resized(int width, int height) override
    {
        // Override window resized method to update camera projection.
        m_main_camera->update_projection(60.0f, 0.1f, CAMERA_FAR_PLANE, float(m_width) / float(m_height));
        m_debug_camera->update_projection(60.0f, 0.1f, CAMERA_FAR_PLANE * 2.0f, float(m_width) / float(m_height));
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void key_pressed(int code) override
    {
        // Handle forward movement.
        if (code == GLFW_KEY_W)
            m_heading_speed = m_camera_speed;
        else if (code == GLFW_KEY_S)
            m_heading_speed = -m_camera_speed;

        // Handle sideways movement.
        if (code == GLFW_KEY_A)
            m_sideways_speed = -m_camera_speed;
        else if (code == GLFW_KEY_D)
            m_sideways_speed = m_camera_speed;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void key_released(int code) override
    {
        // Handle forward movement.
        if (code == GLFW_KEY_W || code == GLFW_KEY_S)
            m_heading_speed = 0.0f;

        // Handle sideways movement.
        if (code == GLFW_KEY_A || code == GLFW_KEY_D)
            m_sideways_speed = 0.0f;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void mouse_pressed(int code) override
    {
        // Enable mouse look.
        if (code == GLFW_MOUSE_BUTTON_RIGHT)
            m_mouse_look = true;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void mouse_released(int code) override
    {
        // Disable mouse look.
        if (code == GLFW_MOUSE_BUTTON_RIGHT)
            m_mouse_look = false;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

protected:
    // -----------------------------------------------------------------------------------------------------------------------------------

    dw::AppSettings intial_app_settings() override
    {
        dw::AppSettings settings;

        settings.resizable    = true;
        settings.maximized    = false;
        settings.refresh_rate = 60;
        settings.major_ver    = 4;
        settings.width        = 1920;
        settings.height       = 1080;
        settings.title        = "Active Ragdoll - Dihara Wijetunga";

        return settings;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

private:
    // -----------------------------------------------------------------------------------------------------------------------------------

    void initialize_physics()
    {
        m_foundation = PxCreateFoundation(PX_PHYSICS_VERSION, m_allocator, m_error_callback);
        m_physics    = PxCreatePhysics(PX_PHYSICS_VERSION, *m_foundation, PxTolerancesScale(), true, nullptr);

        PxSceneDesc scene_desc(m_physics->getTolerancesScale());
        scene_desc.gravity = PxVec3(0.0f, -9.81f, 0.0f);

        m_dispatcher = PxDefaultCpuDispatcherCreate(2);

        scene_desc.cpuDispatcher = m_dispatcher;
        scene_desc.filterShader  = PxDefaultSimulationFilterShader;

        m_scene = m_physics->createScene(scene_desc);

        m_material = m_physics->createMaterial(0.5f, 0.5f, 0.6f);

        PxRigidStatic* ground_plane = PxCreatePlane(*m_physics, PxPlane(0, 1, 0, 20.0f), *m_material);
        m_scene->addActor(*ground_plane);

        m_scene->setVisualizationParameter(PxVisualizationParameter::eSCALE, 1.0f);
        m_scene->setVisualizationParameter(PxVisualizationParameter::eCONTACT_POINT, 1.0f);
        m_scene->setVisualizationParameter(PxVisualizationParameter::eCONTACT_NORMAL, 1.0f);
        m_scene->setVisualizationParameter(PxVisualizationParameter::eCONTACT_ERROR, 1.0f);
        m_scene->setVisualizationParameter(PxVisualizationParameter::eCONTACT_FORCE, 1.0f);
        m_scene->setVisualizationParameter(PxVisualizationParameter::eACTOR_AXES, 1.0f);
        m_scene->setVisualizationParameter(PxVisualizationParameter::eCOLLISION_SHAPES, 1.0f);
        m_scene->setVisualizationParameter(PxVisualizationParameter::eJOINT_LOCAL_FRAMES, 5.0f * m_scale);
        m_scene->setVisualizationParameter(PxVisualizationParameter::eJOINT_LIMITS, 5.0f * m_scale);
        m_scene->setVisualizationParameter(PxVisualizationParameter::eCOLLISION_AXES, 1.0f);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void finish_body(physx::PxRigidDynamic* dyn, physx::PxReal density, physx::PxReal inertiaScale)
    {
        //dyn->setSolverIterationCounts(DEFAULT_SOLVER_ITERATIONS);
        dyn->setMaxDepenetrationVelocity(2.f);
        physx::PxRigidBodyExt::updateMassAndInertia(*dyn, density);
        dyn->setMassSpaceInertiaTensor(dyn->getMassSpaceInertiaTensor() * inertiaScale);
        dyn->setAngularDamping(0.15f);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void config_d6_joint(physx::PxReal swing0, physx::PxReal swing1, physx::PxReal twistLo, physx::PxReal twistHi, physx::PxD6Joint* joint)
    {
        joint->setMotion(physx::PxD6Axis::eSWING1, physx::PxD6Motion::eLIMITED);
        joint->setMotion(physx::PxD6Axis::eSWING2, physx::PxD6Motion::eLIMITED);
        joint->setMotion(physx::PxD6Axis::eTWIST, physx::PxD6Motion::eLIMITED);

        joint->setSwingLimit(physx::PxJointLimitCone(swing0, swing1));
        joint->setTwistLimit(physx::PxJointAngularLimitPair(twistLo, twistHi));
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_ragdoll()
    {
        Joint* joints = m_skeletal_mesh->skeleton()->joints();

#ifdef RIFLE_MESH
        uint32_t j_head_idx        = m_skeletal_mesh->skeleton()->find_joint_index("head");
        uint32_t j_neck_01_idx     = m_skeletal_mesh->skeleton()->find_joint_index("neck_01");
        uint32_t j_spine_03_idx    = m_skeletal_mesh->skeleton()->find_joint_index("spine_03");
        uint32_t j_spine_02_idx    = m_skeletal_mesh->skeleton()->find_joint_index("spine_02");
        uint32_t j_spine_01_idx    = m_skeletal_mesh->skeleton()->find_joint_index("spine_01");
        uint32_t j_pelvis_idx      = m_skeletal_mesh->skeleton()->find_joint_index("pelvis");
        uint32_t j_thigh_l_idx     = m_skeletal_mesh->skeleton()->find_joint_index("thigh_l");
        uint32_t j_calf_l_idx      = m_skeletal_mesh->skeleton()->find_joint_index("calf_l");
        uint32_t j_foot_l_idx      = m_skeletal_mesh->skeleton()->find_joint_index("foot_l");
        uint32_t j_ball_l_idx      = m_skeletal_mesh->skeleton()->find_joint_index("ball_l");
        uint32_t j_thigh_r_idx     = m_skeletal_mesh->skeleton()->find_joint_index("thigh_r");
        uint32_t j_calf_r_idx      = m_skeletal_mesh->skeleton()->find_joint_index("calf_r");
        uint32_t j_foot_r_idx      = m_skeletal_mesh->skeleton()->find_joint_index("foot_r");
        uint32_t j_ball_r_idx      = m_skeletal_mesh->skeleton()->find_joint_index("ball_r");
        uint32_t j_upperarm_l_idx  = m_skeletal_mesh->skeleton()->find_joint_index("upperarm_l");
        uint32_t j_lowerarm_l_idx  = m_skeletal_mesh->skeleton()->find_joint_index("lowerarm_l");
        uint32_t j_hand_l_idx      = m_skeletal_mesh->skeleton()->find_joint_index("hand_l");
        uint32_t j_middle_01_l_idx = m_skeletal_mesh->skeleton()->find_joint_index("middle_01_l");
        uint32_t j_upperarm_r_idx  = m_skeletal_mesh->skeleton()->find_joint_index("upperarm_r");
        uint32_t j_lowerarm_r_idx  = m_skeletal_mesh->skeleton()->find_joint_index("lowerarm_r");
        uint32_t j_hand_r_idx      = m_skeletal_mesh->skeleton()->find_joint_index("hand_r");
        uint32_t j_middle_01_r_idx = m_skeletal_mesh->skeleton()->find_joint_index("middle_01_r");
#else
        uint32_t j_head_idx     = m_skeletal_mesh->skeleton()->find_joint_index("Head");
        uint32_t j_neck_01_idx  = m_skeletal_mesh->skeleton()->find_joint_index("Neck");
        uint32_t j_spine_03_idx = m_skeletal_mesh->skeleton()->find_joint_index("Spine2");
        uint32_t j_spine_02_idx = m_skeletal_mesh->skeleton()->find_joint_index("Spine1");
        uint32_t j_spine_01_idx = m_skeletal_mesh->skeleton()->find_joint_index("Spine");
        uint32_t j_pelvis_idx   = m_skeletal_mesh->skeleton()->find_joint_index("Hips");

        uint32_t j_thigh_l_idx = m_skeletal_mesh->skeleton()->find_joint_index("LeftUpLeg");
        uint32_t j_calf_l_idx  = m_skeletal_mesh->skeleton()->find_joint_index("LeftLeg");
        uint32_t j_foot_l_idx  = m_skeletal_mesh->skeleton()->find_joint_index("LeftFoot");
        uint32_t j_ball_l_idx  = m_skeletal_mesh->skeleton()->find_joint_index("LeftToeBase");

        uint32_t j_thigh_r_idx = m_skeletal_mesh->skeleton()->find_joint_index("RightUpLeg");
        uint32_t j_calf_r_idx  = m_skeletal_mesh->skeleton()->find_joint_index("RightLeg");
        uint32_t j_foot_r_idx  = m_skeletal_mesh->skeleton()->find_joint_index("RightFoot");
        uint32_t j_ball_r_idx  = m_skeletal_mesh->skeleton()->find_joint_index("RightToeBase");

        uint32_t j_upperarm_l_idx  = m_skeletal_mesh->skeleton()->find_joint_index("LeftArm");
        uint32_t j_lowerarm_l_idx  = m_skeletal_mesh->skeleton()->find_joint_index("LeftForeArm");
        uint32_t j_hand_l_idx      = m_skeletal_mesh->skeleton()->find_joint_index("LeftHand");
        uint32_t j_middle_01_l_idx = m_skeletal_mesh->skeleton()->find_joint_index("LeftHandMiddle1");

        uint32_t j_upperarm_r_idx  = m_skeletal_mesh->skeleton()->find_joint_index("RightArm");
        uint32_t j_lowerarm_r_idx  = m_skeletal_mesh->skeleton()->find_joint_index("RightForeArm");
        uint32_t j_hand_r_idx      = m_skeletal_mesh->skeleton()->find_joint_index("RightHand");
        uint32_t j_middle_01_r_idx = m_skeletal_mesh->skeleton()->find_joint_index("RightHandMiddle1");
#endif

        m_ragdoll.m_rigid_bodies.resize(m_skeletal_mesh->skeleton()->num_bones());
        m_ragdoll.m_relative_joint_pos.resize(m_skeletal_mesh->skeleton()->num_bones());
        m_ragdoll.m_original_body_rotations.resize(m_skeletal_mesh->skeleton()->num_bones());
        m_ragdoll.m_body_pos_relative_to_joint.resize(m_skeletal_mesh->skeleton()->num_bones());
        m_ragdoll.m_original_joint_rotations.resize(m_skeletal_mesh->skeleton()->num_bones());

        for (int i = 0; i < m_skeletal_mesh->skeleton()->num_bones(); i++)
            m_ragdoll.m_rigid_bodies[i] = nullptr;

        // ---------------------------------------------------------------------------------------------------------------
        // Create rigid bodies for limbs
        // ---------------------------------------------------------------------------------------------------------------

        float     r   = 5.0f * m_scale;
        glm::mat4 rot = glm::mat4(1.0f);
#ifndef RIFLE_MESH
        rot = glm::rotate(rot, PxHalfPi, glm::vec3(0.0f, 0.0f, 1.0f));
#endif
        PxRigidDynamic* pelvis = create_capsule_bone(j_pelvis_idx, j_neck_01_idx, m_ragdoll, 15.0f * m_scale, rot);
        PxRigidDynamic* head   = create_capsule_bone(j_head_idx, m_ragdoll, glm::vec3(0.0f, 9.0f * m_scale, 0.0f), 4.0f * m_scale, 6.0f * m_scale, rot);

        PxRigidDynamic* l_leg = create_capsule_bone(j_thigh_l_idx, j_calf_l_idx, m_ragdoll, r, rot);
        PxRigidDynamic* r_leg = create_capsule_bone(j_thigh_r_idx, j_calf_r_idx, m_ragdoll, r, rot);

        PxRigidDynamic* l_calf = create_capsule_bone(j_calf_l_idx, j_foot_l_idx, m_ragdoll, r, rot);
        PxRigidDynamic* r_calf = create_capsule_bone(j_calf_r_idx, j_foot_r_idx, m_ragdoll, r, rot);

        PxRigidDynamic* l_arm = create_capsule_bone(j_upperarm_l_idx, j_lowerarm_l_idx, m_ragdoll, r);
        PxRigidDynamic* r_arm = create_capsule_bone(j_upperarm_r_idx, j_lowerarm_r_idx, m_ragdoll, r);

        PxRigidDynamic* l_forearm = create_capsule_bone(j_lowerarm_l_idx, j_hand_l_idx, m_ragdoll, r);
        PxRigidDynamic* r_forearm = create_capsule_bone(j_lowerarm_r_idx, j_hand_r_idx, m_ragdoll, r);

        PxRigidDynamic* l_hand = create_sphere_bone(j_middle_01_l_idx, m_ragdoll, r);
        PxRigidDynamic* r_hand = create_sphere_bone(j_middle_01_r_idx, m_ragdoll, r);

        rot = glm::mat4(1.0f);
#ifndef RIFLE_MESH
        rot = glm::rotate(rot, PxHalfPi, glm::vec3(0.0f, 1.0f, 0.0f));
#endif

        PxRigidDynamic* l_foot = create_capsule_bone(j_foot_l_idx, j_ball_l_idx, m_ragdoll, r, rot);
        PxRigidDynamic* r_foot = create_capsule_bone(j_foot_r_idx, j_ball_r_idx, m_ragdoll, r, rot);

        for (int i = 0; i < m_skeletal_mesh->skeleton()->num_bones(); i++)
        {
            uint32_t        chosen_idx;
            PxRigidDynamic* body = m_ragdoll.find_recent_body(i, m_skeletal_mesh->skeleton(), chosen_idx);

            glm::mat4 body_global_transform     = to_mat4(body->getGlobalPose());
            glm::mat4 inv_body_global_transform = glm::inverse(body_global_transform);
            glm::mat4 bind_pose_ws              = m_model_without_scale * glm::inverse(joints[i].inverse_bind_pose);
            glm::vec3 joint_pos_ws              = joints[i].bind_pos_ws(m_model);

            glm::vec4 p                            = inv_body_global_transform * glm::vec4(joint_pos_ws, 1.0f);
            m_ragdoll.m_relative_joint_pos[i]      = glm::vec3(p.x, p.y, p.z);
            m_ragdoll.m_original_body_rotations[i] = glm::quat_cast(body_global_transform);

            if (m_ragdoll.m_rigid_bodies[i])
            {
                // Rigid body position relative to the joint
                glm::mat4 m = glm::inverse(m_model * glm::inverse(joints[i].inverse_bind_pose));
                p           = m * glm::vec4(to_vec3(m_ragdoll.m_rigid_bodies[i]->getGlobalPose().p), 1.0f);

                m_ragdoll.m_body_pos_relative_to_joint[i] = glm::vec3(p.x, p.y, p.z);
                m_ragdoll.m_original_joint_rotations[i]   = glm::quat_cast(bind_pose_ws);
            }
        }

        // ---------------------------------------------------------------------------------------------------------------
        // Add rigid bodies to scene
        // ---------------------------------------------------------------------------------------------------------------

        // Chest and Head
        m_scene->addActor(*pelvis);
        m_scene->addActor(*head);

        // Left Leg
        m_scene->addActor(*l_leg);
        m_scene->addActor(*l_calf);
        m_scene->addActor(*l_foot);

        // Right Leg
        m_scene->addActor(*r_leg);
        m_scene->addActor(*r_calf);
        m_scene->addActor(*r_foot);

        // Left Arm
        m_scene->addActor(*l_arm);
        m_scene->addActor(*l_forearm);
        m_scene->addActor(*l_hand);

        // Right Arm
        m_scene->addActor(*r_arm);
        m_scene->addActor(*r_forearm);
        m_scene->addActor(*r_hand);

        // ---------------------------------------------------------------------------------------------------------------
        // Create joints
        // ---------------------------------------------------------------------------------------------------------------

        // Chest and Head
        create_d6_joint(pelvis, head, j_neck_01_idx);

        // Chest to Thighs
        create_d6_joint(pelvis, l_leg, j_thigh_l_idx);
        create_d6_joint(pelvis, r_leg, j_thigh_r_idx);

        // Thighs to Calf
        create_revolute_joint(l_leg, l_calf, j_calf_l_idx);
        create_revolute_joint(r_leg, r_calf, j_calf_r_idx);

        // Calf to Foot
        create_d6_joint(l_calf, l_foot, j_foot_l_idx);
        create_d6_joint(r_calf, r_foot, j_foot_r_idx);

        // Chest to Upperarm
        create_d6_joint(pelvis, l_arm, j_upperarm_l_idx);
        create_d6_joint(pelvis, r_arm, j_upperarm_r_idx);

        glm::mat4 arm_rot = glm::mat4(1.0f);
        arm_rot           = glm::rotate(arm_rot, -PxPi / 2.0f, glm::vec3(0.0f, 0.0f, 1.0f));

        // Upperarm to Lowerman
        create_revolute_joint(l_arm, l_forearm, j_lowerarm_l_idx, arm_rot);

        arm_rot = glm::mat4(1.0f);
        arm_rot = glm::rotate(arm_rot, PxPi / 2.0f, glm::vec3(0.0f, 0.0f, 1.0f));
        create_revolute_joint(r_arm, r_forearm, j_lowerarm_r_idx, arm_rot);

        // Lowerman to Hand
        create_d6_joint(l_forearm, l_hand, j_hand_l_idx);
        create_d6_joint(r_forearm, r_hand, j_hand_r_idx);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    PxRigidDynamic* create_capsule_bone(uint32_t parent_idx, uint32_t child_idx, Ragdoll& ragdoll, float r = 0.5f, glm::mat4 rotation = glm::mat4(1.0f))
    {
        Joint* joints = m_skeletal_mesh->skeleton()->joints();

        glm::vec3 parent_pos = joints[parent_idx].bind_pos_ws(m_model);
        glm::vec3 child_pos  = joints[child_idx].bind_pos_ws(m_model);

        float len = glm::length(parent_pos - child_pos);

        // shorten by 0.1 times to prevent overlap
        len -= len * 0.1f;

        float len_minus_2r = len - 2.0f * r;
        float half_height  = len_minus_2r / 2.0f;

        glm::vec3 body_pos = (parent_pos + child_pos) / 2.0f;

        PxShape* shape = m_physics->createShape(PxCapsuleGeometry(r, half_height), *m_material);

        PxTransform local(to_quat(glm::quat_cast(rotation)));
        shape->setLocalPose(local);

        glm::mat4 inv_bind_pose = glm::mat4(glm::mat3(joints[parent_idx].inverse_bind_pose));
        glm::mat4 bind_pose     = glm::inverse(inv_bind_pose);
        glm::mat4 bind_pose_ws  = m_model_without_scale * bind_pose;

        PxTransform px_transform(to_vec3(body_pos), to_quat(glm::quat_cast(bind_pose_ws)));

        PxRigidDynamic* body = m_physics->createRigidDynamic(px_transform);
        body->setMass(m_mass);

        body->attachShape(*shape);

        m_ragdoll.m_rigid_bodies[parent_idx] = body;

        return body;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    PxRigidDynamic* create_capsule_bone(uint32_t parent_idx, Ragdoll& ragdoll, glm::vec3 offset, float l, float r = 0.5f, glm::mat4 rotation = glm::mat4(1.0f))
    {
        Joint* joints = m_skeletal_mesh->skeleton()->joints();

        glm::vec3 parent_pos = joints[parent_idx].bind_pos_ws(m_model) + offset;

        PxShape* shape = m_physics->createShape(PxCapsuleGeometry(r, l), *m_material);

        PxTransform local(to_quat(glm::quat_cast(rotation)));
        shape->setLocalPose(local);

        glm::mat4 inv_bind_pose = glm::mat4(glm::mat3(joints[parent_idx].inverse_bind_pose));
        glm::mat4 bind_pose     = glm::inverse(inv_bind_pose);
        glm::mat4 bind_pose_ws  = m_model_without_scale * bind_pose;

        PxTransform px_transform(to_vec3(parent_pos), to_quat(glm::quat_cast(bind_pose_ws)));

        PxRigidDynamic* body = m_physics->createRigidDynamic(px_transform);

        body->attachShape(*shape);
        body->setMass(m_mass);

        m_ragdoll.m_rigid_bodies[parent_idx] = body;

        return body;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    PxRigidDynamic* create_sphere_bone(uint32_t parent_idx, Ragdoll& ragdoll, float r)
    {
        Joint*    joints     = m_skeletal_mesh->skeleton()->joints();
        glm::vec3 parent_pos = joints[parent_idx].bind_pos_ws(m_model);

        PxShape* shape = m_physics->createShape(PxSphereGeometry(r), *m_material);

        PxRigidDynamic* body = m_physics->createRigidDynamic(PxTransform(to_vec3(parent_pos)));

        body->attachShape(*shape);
        body->setMass(m_mass);

        m_ragdoll.m_rigid_bodies[parent_idx] = body;

        return body;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_d6_joint(PxRigidDynamic* parent, PxRigidDynamic* child, uint32_t joint_pos)
    {
        Joint*    joints = m_skeletal_mesh->skeleton()->joints();
        glm::vec3 p      = joints[joint_pos].bind_pos_ws(m_model);
        glm::quat q      = glm::quat_cast(glm::inverse(joints[joint_pos].inverse_bind_pose));

        PxD6Joint* joint = PxD6JointCreate(*m_physics,
                                           parent,
                                           parent->getGlobalPose().transformInv(PxTransform(to_vec3(p), to_quat(q))),
                                           child,
                                           child->getGlobalPose().transformInv(PxTransform(to_vec3(p), to_quat(q))));

        joint->setConstraintFlag(PxConstraintFlag::eVISUALIZATION, true);

        config_d6_joint(3.14 / 4.f, 3.14f / 4.f, -3.14f / 8.f, 3.14f / 8.f, joint);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_revolute_joint(PxRigidDynamic* parent, PxRigidDynamic* child, uint32_t joint_pos, glm::mat4 rotation = glm::mat4(1.0f))
    {
        Joint*    joints = m_skeletal_mesh->skeleton()->joints();
        glm::vec3 p      = joints[joint_pos].bind_pos_ws(m_model);

        glm::mat4 parent_pose_inv = glm::inverse(to_mat4(parent->getGlobalPose()));
        glm::mat4 child_pose_inv  = glm::inverse(to_mat4(child->getGlobalPose()));
        glm::mat4 joint_transform = glm::mat4(1.0f);
        joint_transform           = glm::translate(joint_transform, p) * rotation;

        PxRevoluteJoint* joint = PxRevoluteJointCreate(*m_physics,
                                                       parent,
                                                       PxTransform(to_mat44(parent_pose_inv * joint_transform)),
                                                       child,
                                                       PxTransform(to_mat44(child_pose_inv * joint_transform)));

        joint->setConstraintFlag(PxConstraintFlag::eVISUALIZATION, true);

        joint->setLimit(PxJointAngularLimitPair(0.0f, PxPi, 0.01f));
        joint->setRevoluteJointFlag(PxRevoluteJointFlag::eLIMIT_ENABLED, true);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void render_physx_debug()
    {
        const physx::PxRenderBuffer& rb = m_scene->getRenderBuffer();
        for (int i = 0; i < rb.getNbLines(); i++)
        {
            const physx::PxDebugLine& line = rb.getLines()[i];

            glm::vec3 color = glm::vec3(1.0f);

            if (line.color0 == PxDebugColor::eARGB_BLACK)
                color = glm::vec3(0.0f);
            if (line.color0 == PxDebugColor::eARGB_RED)
                color = glm::vec3(1.0f, 0.0f, 0.0f);
            if (line.color0 == PxDebugColor::eARGB_GREEN)
                color = glm::vec3(0.0f, 1.0f, 0.0f);
            if (line.color0 == PxDebugColor::eARGB_BLUE)
                color = glm::vec3(0.0f, 0.0f, 1.0f);
            if (line.color0 == PxDebugColor::eARGB_YELLOW)
                color = glm::vec3(1.0f, 1.0f, 0.0f);
            if (line.color0 == PxDebugColor::eARGB_GREY)
                color = glm::vec3(0.5f, 0.5f, 0.5f);

            m_debug_draw.line(glm::vec3(line.pos0.x, line.pos0.y, line.pos0.z), glm::vec3(line.pos1.x, line.pos1.y, line.pos1.z), color);
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    bool create_shaders()
    {
        // Create general shaders
        m_vs = std::unique_ptr<dw::gl::Shader>(dw::gl::Shader::create_from_file(GL_VERTEX_SHADER, "shader/vs.glsl"));
        m_fs = std::unique_ptr<dw::gl::Shader>(dw::gl::Shader::create_from_file(GL_FRAGMENT_SHADER, "shader/fs.glsl"));

        if (!m_vs || !m_fs)
        {
            DW_LOG_FATAL("Failed to create Shaders");
            return false;
        }

        // Create general shader program
        dw::gl::Shader* shaders[] = { m_vs.get(), m_fs.get() };
        m_program                 = std::make_unique<dw::gl::Program>(2, shaders);

        if (!m_program)
        {
            DW_LOG_FATAL("Failed to create Shader Program");
            return false;
        }

        m_program->uniform_block_binding("u_GlobalUBO", 0);
        m_program->uniform_block_binding("u_ObjectUBO", 1);

        // Create Animation shaders
        m_anim_vs = std::unique_ptr<dw::gl::Shader>(dw::gl::Shader::create_from_file(GL_VERTEX_SHADER, "shader/skinning_vs.glsl"));
        m_anim_fs = std::unique_ptr<dw::gl::Shader>(dw::gl::Shader::create_from_file(GL_FRAGMENT_SHADER, "shader/skinning_fs.glsl"));

        if (!m_anim_vs || !m_anim_fs)
        {
            DW_LOG_FATAL("Failed to create Animation Shaders");
            return false;
        }

        // Create Animation shader program
        dw::gl::Shader* anim_shaders[] = { m_anim_vs.get(), m_anim_fs.get() };
        m_anim_program                 = std::make_unique<dw::gl::Program>(2, anim_shaders);

        if (!m_anim_program)
        {
            DW_LOG_FATAL("Failed to create Animation Shader Program");
            return false;
        }

        m_anim_program->uniform_block_binding("u_GlobalUBO", 0);
        m_anim_program->uniform_block_binding("u_ObjectUBO", 1);
        m_anim_program->uniform_block_binding("u_BoneUBO", 2);

        // Create Bone shaders
        m_bone_vs = std::unique_ptr<dw::gl::Shader>(dw::gl::Shader::create_from_file(GL_VERTEX_SHADER, "shader/bone_vs.glsl"));
        m_bone_fs = std::unique_ptr<dw::gl::Shader>(dw::gl::Shader::create_from_file(GL_FRAGMENT_SHADER, "shader/bone_fs.glsl"));

        if (!m_bone_vs || !m_bone_fs)
        {
            DW_LOG_FATAL("Failed to create Bone Shaders");
            return false;
        }

        // Create Bone shader program
        dw::gl::Shader* bone_shaders[] = { m_bone_vs.get(), m_bone_fs.get() };
        m_bone_program                 = std::make_unique<dw::gl::Program>(2, bone_shaders);

        if (!m_bone_program)
        {
            DW_LOG_FATAL("Failed to create Bone Shader Program");
            return false;
        }

        m_bone_program->uniform_block_binding("u_GlobalUBO", 0);
        m_bone_program->uniform_block_binding("u_ObjectUBO", 1);
        m_bone_program->uniform_block_binding("u_BoneUBO", 2);

        return true;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    bool create_uniform_buffer()
    {
        // Create uniform buffer for object matrix data
        m_object_ubo = std::make_unique<dw::gl::UniformBuffer>(GL_DYNAMIC_DRAW, sizeof(ObjectUniforms));

        // Create uniform buffer for global data
        m_global_ubo = std::make_unique<dw::gl::UniformBuffer>(GL_DYNAMIC_DRAW, sizeof(GlobalUniforms));

        // Create uniform buffer for CSM data
        m_bone_ubo = std::make_unique<dw::gl::UniformBuffer>(GL_DYNAMIC_DRAW, sizeof(PoseTransforms));

        return true;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    bool load_mesh()
    {
#ifdef RIFLE_MESH
        m_skeletal_mesh = std::unique_ptr<SkeletalMesh>(SkeletalMesh::load("mesh/Rifle/Rifle_Idle.fbx"));
#else
        m_skeletal_mesh            = std::unique_ptr<SkeletalMesh>(SkeletalMesh::load("mesh/paladin.fbx"));
#endif

        if (!m_skeletal_mesh)
        {
            DW_LOG_FATAL("Failed to load mesh!");
            return false;
        }

        return true;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    bool load_animations()
    {
#ifdef RIFLE_MESH
        m_walk_animation = std::unique_ptr<Animation>(Animation::load("mesh/Rifle/Rifle_Idle.fbx", m_skeletal_mesh->skeleton()));
#else
        m_walk_animation           = std::unique_ptr<Animation>(Animation::load("mesh/paladin.fbx", m_skeletal_mesh->skeleton()));
#endif

        if (!m_walk_animation)
        {
            DW_LOG_FATAL("Failed to load animation!");
            return false;
        }

        m_walk_sampler    = std::make_unique<AnimSample>(m_skeletal_mesh->skeleton(), m_walk_animation.get());
        m_local_to_global = std::make_unique<AnimLocalToGlobal>(m_skeletal_mesh->skeleton());
        m_offset          = std::make_unique<AnimOffset>(m_skeletal_mesh->skeleton());
        m_ragdoll_pose    = std::make_unique<AnimRagdoll>(m_skeletal_mesh->skeleton());

        return true;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_camera()
    {
        m_main_camera  = std::make_unique<dw::Camera>(60.0f, 0.1f, CAMERA_FAR_PLANE, float(m_width) / float(m_height), glm::vec3(0.0f, 5.0f, 20.0f), glm::vec3(0.0f, 0.0, -1.0f));
        m_debug_camera = std::make_unique<dw::Camera>(60.0f, 0.1f, CAMERA_FAR_PLANE * 2.0f, float(m_width) / float(m_height), glm::vec3(0.0f, 5.0f, 20.0f), glm::vec3(0.0f, 0.0, -1.0f));
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void render_mesh(SkeletalMesh* mesh, const ObjectUniforms& transforms, const PoseTransforms& bones)
    {
        // Bind uniform buffers.
        m_object_ubo->bind_base(1);
        m_bone_ubo->bind_base(2);

        // Bind vertex array.
        mesh->bind_vao();

        for (uint32_t i = 0; i < mesh->num_sub_meshes(); i++)
        {
            SubMesh& submesh = mesh->sub_mesh(i);

            // Issue draw call.
            glDrawElementsBaseVertex(GL_TRIANGLES, submesh.num_indices, GL_UNSIGNED_INT, (void*)(sizeof(unsigned int) * submesh.base_index), submesh.base_vertex);
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void render_skeletal_meshes()
    {
        // Bind shader program.
        m_anim_program->use();

        // Bind uniform buffers.
        m_global_ubo->bind_base(0);

        // Draw meshes.
        render_mesh(m_skeletal_mesh.get(), m_character_transforms, m_pose_transforms);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update_object_uniforms(const ObjectUniforms& transform)
    {
        void* ptr = m_object_ubo->map(GL_WRITE_ONLY);
        memcpy(ptr, &transform, sizeof(ObjectUniforms));
        m_object_ubo->unmap();
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update_global_uniforms(const GlobalUniforms& global)
    {
        void* ptr = m_global_ubo->map(GL_WRITE_ONLY);
        memcpy(ptr, &global, sizeof(GlobalUniforms));
        m_global_ubo->unmap();
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update_bone_uniforms(PoseTransforms* bones)
    {
        void* ptr = m_bone_ubo->map(GL_WRITE_ONLY);
        memcpy(ptr, bones, sizeof(PoseTransforms));
        m_bone_ubo->unmap();
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update_transforms(dw::Camera* camera)
    {
        // Update camera matrices.
        m_global_uniforms.view       = camera->m_view;
        m_global_uniforms.projection = camera->m_projection;

        // Update plane transforms.
        m_plane_transforms.model = glm::mat4(1.0f);

        // Update character transforms.
#ifdef RIFLE_MESH
        m_model_without_scale = glm::rotate(m_plane_transforms.model, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
#endif
        m_model_without_scale        = glm::translate(m_model_without_scale, glm::vec3(0.0f, 0.0f, 0.0f));
        m_model                      = glm::scale(m_model_without_scale, glm::vec3(m_scale));
        m_character_transforms.model = m_model;

        m_model_only_scale = glm::mat4(1.0f);
        m_model_only_scale = glm::scale(m_model_only_scale, glm::vec3(m_scale));
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update_camera()
    {
        dw::Camera* current = m_main_camera.get();

        if (m_debug_mode)
            current = m_debug_camera.get();

        float forward_delta = m_heading_speed * m_delta;
        float right_delta   = m_sideways_speed * m_delta;

        current->set_translation_delta(current->m_forward, forward_delta);
        current->set_translation_delta(current->m_right, right_delta);

        double d = 1 - exp(log(0.5) * m_springness * m_delta_seconds);

        m_camera_x = m_mouse_delta_x * m_camera_sensitivity;
        m_camera_y = m_mouse_delta_y * m_camera_sensitivity;

        if (m_mouse_look)
        {
            // Activate Mouse Look
            current->set_rotatation_delta(glm::vec3((float)(m_camera_y),
                                                    (float)(m_camera_x),
                                                    (float)(0.0f)));
        }
        else
        {
            current->set_rotatation_delta(glm::vec3((float)(0),
                                                    (float)(0),
                                                    (float)(0)));
        }

        current->update();

        update_transforms(current);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void gui()
    {
        visualize_hierarchy(m_skeletal_mesh->skeleton());
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update_animations()
    {
        Pose* rifle_pose = m_walk_sampler->sample(m_delta_seconds);

        PoseTransforms* global_transforms = m_local_to_global->generate_transforms(rifle_pose);

        if (!m_simulate)
        {
            for (int i = 0; i < m_ragdoll.m_rigid_bodies.size(); i++)
            {
                if (m_ragdoll.m_rigid_bodies[i])
                {
                    // World position of the rigid body
                    glm::vec4 temp      = m_model * global_transforms->transforms[i] * glm::vec4(m_ragdoll.m_body_pos_relative_to_joint[i], 1.0f);
                    glm::vec3 world_pos = glm::vec3(temp.x, temp.y, temp.z);

                    // Find the difference between the current and original joint rotation.
                    glm::quat joint_rot = glm::quat_cast(m_model_without_scale * global_transforms->transforms[i]);
                    glm::quat diff_rot  = glm::conjugate(m_ragdoll.m_original_joint_rotations[i]) * joint_rot;

                    // The apply the difference rotation to the original rigid body rotation to get the final rotation.
                    glm::quat rotation = m_ragdoll.m_original_body_rotations[i] * diff_rot;

                    PxTransform px_transform(to_vec3(world_pos), to_quat(rotation));

                    m_ragdoll.m_rigid_bodies[i]->setGlobalPose(px_transform);
                }
            }
        }
        else
            global_transforms = m_ragdoll_pose->apply(&m_ragdoll, m_model_only_scale, m_model_without_scale);

        PoseTransforms* final_transforms = m_offset->offset(global_transforms);

        update_bone_uniforms(final_transforms);
        update_skeleton_debug(m_skeletal_mesh->skeleton(), global_transforms);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update_skeleton_debug(Skeleton* skeleton, PoseTransforms* transforms)
    {
        m_joint_pos.clear();

        Joint* joints = skeleton->joints();

        for (int i = 0; i < skeleton->num_bones(); i++)
        {
            glm::mat4 mat = m_model * transforms->transforms[i];

            m_joint_pos.push_back(glm::vec3(mat[3][0], mat[3][1], mat[3][2]));

            if (m_visualize_axis)
                m_debug_draw.transform(mat);
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void visualize_skeleton(Skeleton* skeleton)
    {
        for (int i = 0; i < m_joint_pos.size(); i++)
        {
            glm::vec3 color = glm::vec3(1.0f, 0.0f, 0.0f);

            if (i == 0)
                color = glm::vec3(0.0f, 0.0f, 1.0f);

            if (m_selected_node == i)
                color = glm::vec3(0.0f, 1.0f, 0.0f);

            m_debug_draw.sphere(m_ui_scale, m_joint_pos[i], color);
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void visualize_bones(Skeleton* skeleton)
    {
        Joint* joints = skeleton->joints();

        for (int i = 0; i < skeleton->num_bones(); i++)
        {
            if (joints[i].parent_index == -1)
                continue;

            m_debug_draw.line(m_joint_pos[i], m_joint_pos[joints[i].parent_index], glm::vec3(0.0f, 1.0f, 0.0f));
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void visualize_hierarchy(Skeleton* skeleton)
    {
        static bool skeleton_window = true;

        ImGui::Begin("Skeletal Animation", &skeleton_window);

        ImGui::Checkbox("Visualize Mesh", &m_visualize_mesh);
        ImGui::Checkbox("Visualize Joints", &m_visualize_joints);
        ImGui::Checkbox("Visualize Bones", &m_visualize_bones);
        ImGui::Checkbox("Visualize Bone Axis", &m_visualize_axis);
        ImGui::Checkbox("Pause Simulation", &m_pause_sim);

        if (ImGui::Checkbox("Simulate", &m_simulate))
            m_ragdoll.set_kinematic(!m_simulate);

        ImGui::Checkbox("PhysX Debug Draw", &m_physx_debug_draw);

        ImGui::Separator();

        bool is_combo_open = false;

        for (int i = 0; i < 20; i++)
        {
            if (ImGui::BeginCombo(kJointNames[i].c_str(), m_joint_names[m_selected_joints[i]].c_str()))
            {
                is_combo_open = true;

                for (int j = 0; j < m_joint_names.size(); j++)
                {
                    bool is_selected = (j == m_selected_joints[i]);

                    if (ImGui::Selectable(m_joint_names[j].c_str(), is_selected))
                        m_selected_joints[i] = j;

                    if (ImGui::IsItemHovered())
                    {
                        m_selected_node    = j - 1;
                        m_visualize_joints = true;
                    }

                    if (is_selected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        }

        if (!is_combo_open)
            m_selected_node = -1;

        ImGui::Separator();

        ImGui::Text("Hierarchy");

        Joint* joints = skeleton->joints();

        for (int i = 0; i < skeleton->num_bones(); i++)
        {
            if (m_index_stack.size() > 0 && joints[i].parent_index < m_index_stack.back().first)
            {
                while (m_index_stack.back().first != joints[i].parent_index)
                {
                    if (m_index_stack.back().second)
                        ImGui::TreePop();

                    m_index_stack.pop_back();
                }
            }

            bool parent_opened = false;

            for (auto& p : m_index_stack)
            {
                if (p.first == joints[i].parent_index && p.second)
                {
                    parent_opened = true;
                    break;
                }
            }

            if (!parent_opened && m_index_stack.size() > 0)
                continue;

            ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick | (m_selected_node == i ? ImGuiTreeNodeFlags_Selected : 0);
            bool               opened     = ImGui::TreeNodeEx(joints[i].name.c_str(), node_flags);

            if (ImGui::IsItemClicked())
                m_selected_node = i;

            m_index_stack.push_back({ i, opened });
        }

        if (m_index_stack.size() > 0)
        {
            while (m_index_stack.size() > 0)
            {
                if (m_index_stack.back().second)
                    ImGui::TreePop();

                m_index_stack.pop_back();
            }
        }

        ImGui::End();
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

private:
    // General GPU resources.
    std::unique_ptr<dw::gl::Shader>        m_vs;
    std::unique_ptr<dw::gl::Shader>        m_fs;
    std::unique_ptr<dw::gl::Program>       m_program;
    std::unique_ptr<dw::gl::UniformBuffer> m_object_ubo;
    std::unique_ptr<dw::gl::UniformBuffer> m_bone_ubo;
    std::unique_ptr<dw::gl::UniformBuffer> m_global_ubo;

    // Animation shaders.
    std::unique_ptr<dw::gl::Shader>  m_anim_vs;
    std::unique_ptr<dw::gl::Shader>  m_anim_fs;
    std::unique_ptr<dw::gl::Program> m_anim_program;

    // Bone shaders.
    std::unique_ptr<dw::gl::Shader>  m_bone_vs;
    std::unique_ptr<dw::gl::Shader>  m_bone_fs;
    std::unique_ptr<dw::gl::Program> m_bone_program;

    // Camera.
    std::unique_ptr<dw::Camera> m_main_camera;
    std::unique_ptr<dw::Camera> m_debug_camera;

    // Uniforms.
    ObjectUniforms m_plane_transforms;
    ObjectUniforms m_character_transforms;
    GlobalUniforms m_global_uniforms;
    PoseTransforms m_pose_transforms;

    // Animations
    std::unique_ptr<Animation>         m_walk_animation;
    std::unique_ptr<AnimSample>        m_walk_sampler;
    std::unique_ptr<AnimLocalToGlobal> m_local_to_global;
    std::unique_ptr<AnimOffset>        m_offset;
    std::unique_ptr<AnimRagdoll>       m_ragdoll_pose;

    physx::PxDefaultAllocator      m_allocator;
    physx::PxDefaultErrorCallback  m_error_callback;
    physx::PxFoundation*           m_foundation = nullptr;
    physx::PxPhysics*              m_physics    = nullptr;
    physx::PxDefaultCpuDispatcher* m_dispatcher = nullptr;
    physx::PxScene*                m_scene      = nullptr;
    physx::PxMaterial*             m_material   = nullptr;
    glm::mat4                      m_model;
    glm::mat4                      m_model_without_scale = glm::mat4(1.0f);
    glm::mat4                      m_model_only_scale    = glm::mat4(1.0f);
    float                          m_scale               = 0.1f;
    float                          m_ui_scale            = m_scale;
    float                          m_mass                = 1.0f;

    // Mesh
    std::unique_ptr<SkeletalMesh>
        m_skeletal_mesh;

    // Camera controls.
    bool  m_mouse_look         = false;
    bool  m_debug_mode         = false;
    float m_heading_speed      = 0.0f;
    float m_sideways_speed     = 0.0f;
    float m_camera_sensitivity = 0.05f;
    float m_camera_speed       = 0.1f;

    // GUI
    bool m_pause_sim        = false;
    bool m_visualize_mesh   = true;
    bool m_visualize_joints = true;
    bool m_visualize_bones  = true;
    bool m_visualize_axis   = false;

    // Camera orientation.
    float   m_camera_x;
    float   m_camera_y;
    float   m_springness            = 1.0f;
    float   m_blend_factor          = 0.0f;
    float   m_additive_blend_factor = 0.0f;
    float   m_pitch_blend           = 0.0f;
    float   m_yaw_blend             = 0.0f;
    bool    m_simulate              = false;
    bool    m_physx_debug_draw      = false;
    bool    m_first                 = true;
    Ragdoll m_ragdoll;

    int32_t                               m_selected_node = -1;
    std::vector<glm::vec3>                m_joint_pos;
    std::vector<glm::mat4>                m_joint_mat;
    std::vector<std::pair<int32_t, bool>> m_index_stack;

    std::vector<std::string> m_joint_names;
    std::vector<int32_t>     m_selected_joints;
};

DW_DECLARE_MAIN(AnimationStateMachine)
