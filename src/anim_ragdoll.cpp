#include "anim_ragdoll.h"
#include <gtc/matrix_transform.hpp>
#include <iostream>

PxRigidDynamic* Ragdoll::find_recent_body(uint32_t idx, Skeleton* skeleton, uint32_t& chosen_idx)
{
    Joint* joints = skeleton->joints();

    chosen_idx           = idx;
    PxRigidDynamic* body = m_rigid_bodies[idx];

    while (!body)
    {
        idx        = joints[idx].parent_index;
        body       = m_rigid_bodies[idx];
        chosen_idx = idx;
    }

    return body;
}

void Ragdoll::set_kinematic(bool state)
{
    for (int i = 0; i < m_rigid_bodies.size(); i++)
    {
        if (m_rigid_bodies[i])
        {
            m_rigid_bodies[i]->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, state);

            if (!state)
                m_rigid_bodies[i]->wakeUp();
        }
    }
}

glm::vec3 pos_from_transform(glm::mat4 m)
{
    return glm::vec3(m[3][0], m[3][1], m[3][2]);
}

AnimRagdoll::AnimRagdoll(Skeleton* skeleton) :
    m_skeleton(skeleton)
{
}

AnimRagdoll::~AnimRagdoll()
{
}

PoseTransforms* AnimRagdoll::apply(Ragdoll* ragdoll, glm::mat4 model_scale, glm::mat4 model_rotation)
{
    if (ragdoll->m_rigid_bodies.size() > 0)
    {
        Joint* joints = m_skeleton->joints();

        for (uint32_t i = 0; i < m_skeleton->num_bones(); i++)
        {
            uint32_t        chosen_idx;
            PxRigidDynamic* body = ragdoll->find_recent_body(i, m_skeleton, chosen_idx);

            glm::mat4 global_transform = to_mat4(body->getGlobalPose());
            glm::vec4 global_joint_pos = global_transform * glm::vec4(ragdoll->m_relative_joint_pos[i], 1.0f);

            glm::quat body_rot = glm::quat_cast(global_transform);
            glm::quat diff_rot = glm::conjugate(ragdoll->m_original_body_rotations[i]) * body_rot;

            glm::mat4 translation = glm::mat4(1.0f);
            translation           = glm::translate(translation, glm::vec3(global_joint_pos.x, global_joint_pos.y, global_joint_pos.z));

            glm::quat final_rotation = joints[i].original_rotation * diff_rot;
            glm::mat4 rotation       = model_rotation * glm::mat4_cast(final_rotation);

            glm::mat4 model = model_rotation * model_scale;

            m_transforms.transforms[i] = glm::inverse(model) * translation * rotation * model_scale;
        }
    }

    return &m_transforms;
}