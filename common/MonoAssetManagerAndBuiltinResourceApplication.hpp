// Copyright (C) 2023-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXAMPLES_COMMON_MONO_ASSET_MANAGER_AND_BUILTIN_RESOURCE_APPLICATION_HPP_INCLUDED_
#define _NBL_EXAMPLES_COMMON_MONO_ASSET_MANAGER_AND_BUILTIN_RESOURCE_APPLICATION_HPP_INCLUDED_

// we need a system and a logger
#include "../common/MonoSystemMonoLoggerApplication.hpp"
#ifdef NBL_EMBED_BUILTIN_RESOURCES
#include "nbl/this_example/builtin/CArchive.h"
#endif

#include "nbl\system\CMountDirectoryArchive.h"

namespace nbl::examples
{

// Virtual Inheritance because apps might end up doing diamond inheritance
class MonoAssetManagerAndBuiltinResourceApplication : public virtual MonoSystemMonoLoggerApplication
{
		using base_t = MonoSystemMonoLoggerApplication;

	public:
		using base_t::base_t;

	protected:
		// need this one for skipping passing all args into ApplicationFramework
		MonoAssetManagerAndBuiltinResourceApplication() = default;

		virtual bool onAppInitialized(core::smart_refctd_ptr<system::ISystem>&& system) override
		{
			if (!base_t::onAppInitialized(std::move(system)))
				return false;

			using namespace core;
			m_assetMgr = make_smart_refctd_ptr<asset::IAssetManager>(smart_refctd_ptr(m_system));

		#ifdef NBL_EMBED_BUILTIN_RESOURCES
			m_system->mount(make_smart_refctd_ptr<nbl::this_example::builtin::CArchive>(smart_refctd_ptr(m_logger)),"app_resources");
		#else
			m_system->mount(make_smart_refctd_ptr<system::CMountDirectoryArchive>(localInputCWD/"app_resources",smart_refctd_ptr(m_logger),m_system.get()));
		#endif

			return true;
		}

		core::smart_refctd_ptr<asset::IAssetManager> m_assetMgr;
};

}

#endif // _CAMERA_IMPL_