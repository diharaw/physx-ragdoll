#include "anim_ragdoll.h"

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

glm::vec3 pos_from_transform(glm::mat4 m) {
    return glm::vec3(m[3][0], m[3][1], m[3][2]);
}

AnimRagdoll::AnimRagdoll(Skeleton* skeleton) :
    m_skeleton(skeleton)
{
}

AnimRagdoll::~AnimRagdoll()
{
}

PoseTransforms* AnimRagdoll::apply(Ragdoll* ragdoll, glm::mat4 model)
{
    if (ragdoll->m_rigid_bodies.size() > 0)
    {
        Joint* joints = m_skeleton->joints();

        for (uint32_t i = 0; i < m_skeleton->num_bones(); i++)
        {
            uint32_t        chosen_idx;
            PxRigidDynamic* body = ragdoll->find_recent_body(i, m_skeleton, chosen_idx);

            m_transforms.transforms[i] = to_mat4(body->getGlobalPose()) * ragdoll->m_body_to_joint_transform[i];
        }
    }

    return &m_transforms;
}