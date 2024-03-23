// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


// I've moved out a tiny part of this example into a shared header for reuse, please open and read it.
#include "../common/SimpleWindowedApplication.hpp"
#include <nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp>

#include <nbl/builtin/hlsl/central_limit_blur/common.hlsl>
#include "app_resources/descriptors.hlsl"

#include "CArchive.h"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;

#define _NBL_PLATFORM_WINDOWS_

class BoxBlurDemo final : public examples::SimpleWindowedApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
	using base_t = examples::SimpleWindowedApplication;
	using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;
	using clock_t = std::chrono::steady_clock;

	constexpr static inline clock_t::duration DisplayImageDuration = std::chrono::milliseconds( 900 );

public:
	BoxBlurDemo( const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD 
	) : system::IApplicationFramework( _localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD ) {}
	
	bool onAppInitialized( smart_refctd_ptr<ISystem>&& system ) override
	{
		// Remember to call the base class initialization!
		if( !base_t::onAppInitialized( core::smart_refctd_ptr( system ) ) )
		{
			return false;
		}
		if( !asset_base_t::onAppInitialized( std::move( system ) ) )
		{
			return false;
		}

		
		auto checkedLoad = [ & ]<class T>( const char* filePath ) -> smart_refctd_ptr<T>
		{
			IAssetLoader::SAssetLoadParams lparams = {};
			lparams.logger = m_logger.get();
			lparams.workingDirectory = "";
			// The `IAssetManager::getAsset` function is very complex, in essencee it:
			// 1. takes a cache key or an IFile, if you gave it an `IFile` skip to step 3
			// 2. it consults the loader override about how to get an `IFile` from your cache key
			// 3. handles any failure in opening an `IFile` (which is why it takes a supposed filename), it allows the override to give a different file
			// 4. tries to derive a working directory if you haven't provided one
			// 5. looks for the assets in the cache if you haven't disabled that in the loader parameters
			// 5a. lets the override choose relevant assets from the ones found under the cache key
			// 5b. if nothing was found it lets the override intervene one last time
			// 6. if there's no file to load from, return no assets
			// 7. try all loaders associated with a file extension
			// 8. then try all loaders by opening the file and checking if it will load
			// 9. insert loaded assets into cache if required
			// 10. restore assets from dummy state if needed (more on that in other examples)
			// Take the docs with a grain of salt, the `getAsset` will be rewritten to deal with restores better in the near future.
			nbl::asset::SAssetBundle bundle = m_assetMgr->getAsset( filePath, lparams );
			if( bundle.getContents().empty() )
			{
				m_logger->log( "Asset %s failed to load! Are you sure it exists?", ILogger::ELL_ERROR, filePath );
				return nullptr;
			}
			// All assets derive from `nbl::asset::IAsset`, and can be casted down if the type matches
			static_assert( std::is_base_of_v<nbl::asset::IAsset, T> );
			// The type of the root assets in the bundle is not known until runtime, so this is kinda like a `dynamic_cast` which will return nullptr on type mismatch
			auto typedAsset = IAsset::castDown<T>( bundle.getContents()[ 0 ] ); // just grab the first asset in the bundle
			if( !typedAsset )
			{
				m_logger->log( "Asset type mismatch want %d got %d !", ILogger::ELL_ERROR, T::AssetType, bundle.getAssetType() );

			}
			return typedAsset;
		};

		auto textureToBlur = checkedLoad.operator()< nbl::asset::ICPUImage >( "app_resources/tex.jpg" );
		if( !textureToBlur )
		{
			return logFail( "Failed to load texture!\n" );
		}
		const auto& inCpuTexInfo = textureToBlur->getCreationParameters();
		
		auto createGPUImages = [ & ]( core::bitflag<IGPUImage::E_USAGE_FLAGS> usageFlags, asset::E_FORMAT format, std::string_view name 
									  ) -> smart_refctd_ptr<nbl::video::IGPUImage> {
			video::IGPUImage::SCreationParams gpuImageCreateInfo;
			gpuImageCreateInfo.flags = inCpuTexInfo.flags | IImage::ECF_MUTABLE_FORMAT_BIT;
			gpuImageCreateInfo.type = inCpuTexInfo.type;
			gpuImageCreateInfo.extent = inCpuTexInfo.extent;
			gpuImageCreateInfo.mipLevels = inCpuTexInfo.mipLevels;
			gpuImageCreateInfo.arrayLayers = inCpuTexInfo.arrayLayers;
			gpuImageCreateInfo.samples = inCpuTexInfo.samples;
			gpuImageCreateInfo.tiling = video::IGPUImage::TILING::OPTIMAL;
			gpuImageCreateInfo.usage = usageFlags | asset::IImage::EUF_TRANSFER_DST_BIT;
			gpuImageCreateInfo.queueFamilyIndexCount = 0u;
			gpuImageCreateInfo.queueFamilyIndices = nullptr;

			gpuImageCreateInfo.format = //format;
				m_physicalDevice->promoteImageFormat({ inCpuTexInfo.format, gpuImageCreateInfo.usage }, gpuImageCreateInfo.tiling );
			//gpuImageCreateInfo.viewFormats.set( E_FORMAT::EF_R8G8B8A8_SRGB );
			//gpuImageCreateInfo.viewFormats.set( E_FORMAT::EF_R8G8B8A8_UNORM );
			auto gpuImage = m_device->createImage( std::move( gpuImageCreateInfo ) );

			auto gpuImageMemReqs = gpuImage->getMemoryReqs();
			gpuImageMemReqs.memoryTypeBits &= m_physicalDevice->getDeviceLocalMemoryTypeBits();
			m_device->allocate( gpuImageMemReqs, gpuImage.get(), video::IDeviceMemoryAllocation::EMAF_NONE );

			gpuImage->setObjectDebugName( name.data() );
			return gpuImage;
		};
		smart_refctd_ptr<nbl::video::IGPUImage> gpuImg = createGPUImages( 
			IImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT /* | IImage::E_USAGE_FLAGS::EUF_STORAGE_BIT */, E_FORMAT::EF_R8G8B8A8_SRGB, "GPU Image");
		const auto& gpuImgParams = gpuImg->getCreationParameters();

		smart_refctd_ptr<nbl::video::IGPUImageView> sampledView;
		smart_refctd_ptr<nbl::video::IGPUImageView> unormView;
		{
			sampledView = m_device->createImageView( {
				.flags = IGPUImageView::ECF_NONE,
				.subUsages = IImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT,
				.image = gpuImg,
				.viewType = IGPUImageView::ET_2D,
				.format = E_FORMAT::EF_R8G8B8A8_SRGB
			} );
			sampledView->setObjectDebugName( "Sampled sRGB view" );

			unormView = m_device->createImageView( {
				.flags = IGPUImageView::ECF_NONE,
				.subUsages = IImage::E_USAGE_FLAGS::EUF_STORAGE_BIT,
				.image = gpuImg,
				.viewType = IGPUImageView::ET_2D,
				.format = E_FORMAT::EF_R8G8B8A8_UNORM
			} );
			unormView->setObjectDebugName( "UNORM view" );
		}
		assert( gpuImg && sampledView && unormView );


		constexpr uint32_t WorkgroupSize = 256; // TODO: Number of Passes as parameter
		smart_refctd_ptr<IGPUShader> shader;
		{
			auto computeMain = checkedLoad.operator() < nbl::asset::ICPUShader > ( "app_resources/main.comp.hlsl" );
			smart_refctd_ptr<ICPUShader> overridenUnspecialized = CHLSLCompiler::createOverridenCopy(
				computeMain.get(), "#define WORKGROUP_SIZE %s\n", std::to_string( WorkgroupSize ).c_str() );
			shader = m_device->createShader( overridenUnspecialized.get() );
			if( !shader )
			{
				return logFail( "Creation of a GPU Shader to from CPU Shader source failed!" );
			}
		}
		

		// Now surface indep resources
		m_semaphore = m_device->createSemaphore( m_submitIx );
		if( !m_semaphore )
		{
			return logFail( "Failed to Create a Semaphore!" );
		}

		smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout;
		{
			NBL_CONSTEXPR_STATIC nbl::video::IGPUDescriptorSetLayout::SBinding bindings[] = {
			{
				.binding = inputViewBinding,
				.type = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
				.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags = IShader::ESS_COMPUTE,
				.count = 1,
				.samplers = nullptr
			},
			{
				.binding = outputViewBinding,
				.type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
				.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags = IShader::ESS_COMPUTE,
				.count = 1,
				.samplers = nullptr
			}
			};
			dsLayout = m_device->createDescriptorSetLayout( bindings );
			if( !dsLayout )
			{
				return logFail( "Failed to create a Descriptor Layout!\n" );
			}
		}
		
		ISwapchain::SCreationParams swapchainParams = { .surface = m_surface->getSurface() };
		// Need to choose a surface format
		if( !swapchainParams.deduceFormat( m_physicalDevice ) )
		{
			return logFail( "Could not choose a Surface Format for the Swapchain!" );
		}
		
		// Let's just use the same queue since there's no need for async present
		if( !m_surface || !m_surface->init( getGraphicsQueue(), 0, swapchainParams.sharedParams ) )
		{
			return logFail( "Could not create Window & Surface or initialize the Surface!" );
		}
		m_maxFramesInFlight = m_surface->getMaxFramesInFlight();

		{
			const uint32_t setCount = m_maxFramesInFlight;
			auto pool = m_device->createDescriptorPoolForDSLayouts( IDescriptorPool::E_CREATE_FLAGS::ECF_NONE, { &dsLayout.get(),1 }, &setCount );
			if( !pool )
			{
				return logFail( "Failed to Create Descriptor Pool" );
			}
				
			for( uint64_t i = 0u; i < m_maxFramesInFlight; ++i )
			{
				m_descriptorSets[ i ] = pool->createDescriptorSet( core::smart_refctd_ptr( dsLayout ) );
				if( !m_descriptorSets[ i ] )
				{
					return logFail( "Could not create Descriptor Set!" );
				}
			}
		}




		const asset::SPushConstantRange pushConst[] = { {.stageFlags = IShader::ESS_COMPUTE, .offset = 0, .size = sizeof( nbl::hlsl::central_limit_blur::BoxBlurParams )} };
		smart_refctd_ptr<nbl::video::IGPUPipelineLayout> pplnLayout = m_device->createPipelineLayout( pushConst, smart_refctd_ptr(dsLayout));
		if( !pplnLayout )
		{
			return logFail( "Failed to create a Pipeline Layout!\n" );
		}

		smart_refctd_ptr<nbl::video::IGPUComputePipeline> pipeline;
		{
			IGPUComputePipeline::SCreationParams params = {};
			params.layout = pplnLayout.get();
			params.shader.entryPoint = "main";
			params.shader.shader = shader.get();
			if( !m_device->createComputePipelines( nullptr, { &params, 1 }, &pipeline ) )
			{
				return logFail( "Failed to create pipelines (compile & link shaders)!\n" );
			}
		}



		smart_refctd_ptr<video::IGPUSampler> sampler = m_device->createSampler( {} );
		smart_refctd_ptr<nbl::video::IDescriptorPool> pool = m_device->createDescriptorPoolForDSLayouts( IDescriptorPool::ECF_NONE, { &dsLayout.get(),1 } );
		smart_refctd_ptr<nbl::video::IGPUDescriptorSet> ds = pool->createDescriptorSet( std::move( dsLayout ) );
		{
			// Views must be in the same layout because we read from them simultaneously 
			IGPUDescriptorSet::SDescriptorInfo info[ 2 ];
			info[ 0 ].desc = sampledView;
			info[ 0 ].info.image = { .sampler = sampler, .imageLayout = IImage::LAYOUT::GENERAL };
			info[ 1 ].desc = unormView;
			info[ 1 ].info.image = { .sampler = nullptr, .imageLayout = IImage::LAYOUT::GENERAL };

			IGPUDescriptorSet::SWriteDescriptorSet writes[] = {
				{ .dstSet = ds.get(), .binding = inputViewBinding, .arrayElement = 0, .count = 1, .info = &info[ 0 ] },
				{ .dstSet = ds.get(), .binding = outputViewBinding, .arrayElement = 0, .count = 1, .info = &info[ 1 ] },
			};
			const bool success = m_device->updateDescriptorSets( writes, {} );
			assert( success );
		}

		ds->setObjectDebugName( "Box blur DS" );
		pplnLayout->setObjectDebugName( "Box Blur PPLN Layout" );



		// Transfer stage
		const bool needsOwnershipTransfer = getTransferUpQueue()->getFamilyIndex()!=getComputeQueue()->getFamilyIndex();
		auto transferSema = m_device->createSemaphore(0);
		IQueue::SSubmitInfo::SSemaphoreInfo transferDone[] = {
			{.semaphore = transferSema.get(),.value = 1,.stageMask = PIPELINE_STAGE_FLAGS::COPY_BIT} };
		{
			IQueue* queue = getTransferUpQueue();

			smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdbuf;
			smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(
				queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT );
			if( !cmdpool->createCommandBuffers( IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &cmdbuf ) )
			{
				return logFail( "Failed to create Command Buffers!\n" );
			}

			
			cmdbuf->begin( IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT );

			const IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> imgLayouts[] = {
				{
					.barrier = {
						.dep={
							// there's no need for a source synchronization because Host Ops become available and visible pre-submit
							.srcStageMask = PIPELINE_STAGE_FLAGS::NONE, .srcAccessMask = ACCESS_FLAGS::NONE,
							.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT, .dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT
						},
					},
					.image = gpuImg.get(),
					.subresourceRange = { .aspectMask = IImage::EAF_COLOR_BIT, .levelCount = 1, .layerCount = 1 },
					.oldLayout = IImage::LAYOUT::UNDEFINED,
					.newLayout = IImage::LAYOUT::TRANSFER_DST_OPTIMAL,
				}
			};
			if( !cmdbuf->pipelineBarrier( nbl::asset::EDF_NONE, { .imgBarriers = imgLayouts } ) )
			{
				return logFail( "Failed to issue barrier!\n" );
			}

			queue->startCapture();
			IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[] = { {.cmdbuf = cmdbuf.get()} };
			SIntendedSubmitInfo intendedSubmit = { .frontHalf = {.queue = queue, .waitSemaphores = {/*wait for no - one*/}, 
				.commandBuffers = cmdbufs }, .signalSemaphores = transferDone };

			bool uploaded = m_utils->updateImageViaStagingBuffer( intendedSubmit, textureToBlur->getBuffer(), inCpuTexInfo.format,
				gpuImg.get(), IImage::LAYOUT::TRANSFER_DST_OPTIMAL, textureToBlur->getRegions()
			);
			if( !uploaded )
			{
				return logFail( "Failed to upload cpu tex!\n" );
			}

			const IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> releaseOwnership[] = {
				{
					.barrier = {
						.dep = {
							.srcStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT, .srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT,
							// there's no need for a source synchronization scope because the Submit implicit
							// Timeline Semaphore guarantees already sync us and make our writes available
						},
						.ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::RELEASE,
						.otherQueueFamilyIndex = needsOwnershipTransfer ? getComputeQueue()->getFamilyIndex():IQueue::FamilyIgnored
					},
					.image = gpuImg.get(),
					.subresourceRange = {.aspectMask = IImage::EAF_COLOR_BIT, .levelCount = 1, .layerCount = 1 },
					.oldLayout = IImage::LAYOUT::TRANSFER_DST_OPTIMAL,
				    .newLayout = IImage::LAYOUT::GENERAL,
				}
			};
			if( !cmdbuf->pipelineBarrier( nbl::asset::EDF_NONE, { .imgBarriers = releaseOwnership } ) )
			{
				return logFail( "Failed to issue barrier!\n" );
			}

			cmdbuf->end();
			const IQueue::SSubmitInfo info = intendedSubmit;
			queue->submit({&info,1});
			queue->endCapture();

			// WARNING : Depending on OVerflows, `transferDone->value!=1` so if you want to sync the compute submit against that,
			// use `transferDone` directly as the wait semaphore!
			const ISemaphore::SWaitInfo waitInfo = {transferDone->semaphore,transferDone->value};
			m_device->blockForSemaphores( { &waitInfo,1 } );
		}
		
		constexpr size_t StartedValue = 0;
		constexpr size_t FinishedValue = 45;
		static_assert( StartedValue < FinishedValue );
		smart_refctd_ptr<ISemaphore> progress = m_device->createSemaphore( StartedValue );
		IQueue* queue = getComputeQueue();

		smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdbuf;
		smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(
			queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT );
		if( !cmdpool->createCommandBuffers( IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &cmdbuf ) )
		{
			return logFail( "Failed to create Command Buffers!\n" );
		}

		const uint64_t itemsPerWg = gpuImgParams.extent.width / WorkgroupSize;
		hlsl::central_limit_blur::BoxBlurParams pushConstData = {
			.radius = 4.f,
			.direction = 0,
			.channelCount = nbl::asset::getFormatChannelCount( gpuImgParams.format ),
			.wrapMode = hlsl::central_limit_blur::WrapMode::WRAP_MODE_CLAMP_TO_EDGE,
			.borderColorType = hlsl::central_limit_blur::BorderColor::BORDER_COLOR_FLOAT_OPAQUE_BLACK,
		};

		cmdbuf->begin( IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT );
		cmdbuf->beginDebugMarker( "Box Blur dispatches", core::vectorSIMDf( 0, 1, 0, 1 ) );
		if( needsOwnershipTransfer )
		{
			const IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> imgLayouts[] = {
				// this is only for Ownership Acquire, the transfer queue does the layout xform,
				// so if `!needsOwnershipTransfer` we skip to prevent a double layout transition
				{
					.barrier = {
						// src flags are ignored by Acquire
						.dep = {
							.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
							.dstAccessMask = ACCESS_FLAGS::SAMPLED_READ_BIT | ACCESS_FLAGS::STORAGE_WRITE_BIT,
						},
						.ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::ACQUIRE,
						.otherQueueFamilyIndex =  getTransferUpQueue()->getFamilyIndex()
					},
					.image = gpuImg.get(),
					.subresourceRange = {.aspectMask = IImage::EAF_COLOR_BIT, .levelCount = 1, .layerCount = 1 },
					.oldLayout = IImage::LAYOUT::TRANSFER_DST_OPTIMAL,
					.newLayout = IImage::LAYOUT::GENERAL
				}
			};
			if( !cmdbuf->pipelineBarrier( nbl::asset::EDF_NONE, { .imgBarriers = imgLayouts } ) )
			{
				return logFail( "Failed to issue barrier!\n" );
			}
		}
		cmdbuf->bindComputePipeline( pipeline.get() );
		cmdbuf->bindDescriptorSets( nbl::asset::EPBP_COMPUTE, pplnLayout.get(), 0, 1, &ds.get() );
		cmdbuf->pushConstants( pplnLayout.get(), IShader::ESS_COMPUTE, 0, sizeof( pushConstData ), &pushConstData );
		cmdbuf->dispatch( 1, gpuImgParams.extent.height, 1 );

		const nbl::asset::SMemoryBarrier barriers[] = {
			{
				.srcStageMask = nbl::asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
				.srcAccessMask = nbl::asset::ACCESS_FLAGS::SHADER_WRITE_BITS,
				.dstStageMask = nbl::asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
				.dstAccessMask= nbl::asset::ACCESS_FLAGS::SHADER_READ_BITS,
			}
		};
		// TODO: you don't need a pipeline barrier just before the end of the last command buffer to be submitted
		// Timeline semaphore takes care of all the memory deps between a signal and a wait 
		if( !cmdbuf->pipelineBarrier( nbl::asset::EDF_NONE, { .memBarriers = barriers } ) )
		{
			return logFail( "Failed to issue barrier!\n" );
		}
		//cmdbuf->dispatch( gpuTexSize.width, 1, 1 );
		cmdbuf->endDebugMarker();
		cmdbuf->end();
		
		{
			const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[] = { {.cmdbuf = cmdbuf.get()} };
			const IQueue::SSubmitInfo::SSemaphoreInfo signals[] = { 
				{.semaphore = progress.get(), .value = FinishedValue, .stageMask = asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS} };
			IQueue::SSubmitInfo submitInfos[] = { 
				{ .waitSemaphores = transferDone, .commandBuffers = cmdbufs, .signalSemaphores = signals } };

			// This is super useful for debugging multi-queue workloads and by default RenderDoc delimits captures only by Swapchain presents.
			queue->startCapture();
			queue->submit( submitInfos );
			queue->endCapture();
		}
		const ISemaphore::SWaitInfo waitInfos[] = { { .semaphore = progress.get(), .value = FinishedValue } };
		m_device->blockForSemaphores( waitInfos );

		return true;
	}

	// Platforms like WASM expect the main entry point to periodically return control, hence if you want a crossplatform app, you have to let the framework deal with your "game loop"
	void workLoopBody() override {}

	// Whether to keep invoking the above. In this example because its headless GPU compute, we do all the work in the app initialization.
	bool keepRunning() override { return false; }

	// Just to run destructors in a nice order
	bool onAppTerminated() override
	{
		getGraphicsQueue()->endCapture();
		return base_t::onAppTerminated();
	}

	// Will get called mid-initialization, via `filterDevices` between when the API Connection is created and Physical Device is chosen
	inline core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
	{
		// So let's create our Window and Surface then!
		if( !m_surface )
		{
			{
				ui::IWindow::SCreationParams params = {};
				params.callback = core::make_smart_refctd_ptr<nbl::video::ISimpleManagedSurface::ICallback>();
				params.width = 512;
				params.height = 512;
				params.x = 32;
				params.y = 32;
				// Don't want to have a window lingering about before we're ready so create it hidden.
				// Only programmatic resize, not regular.
				params.flags = ui::IWindow::ECF_HIDDEN | ui::IWindow::ECF_BORDERLESS | ui::IWindow::ECF_RESIZABLE;
				params.windowCaption = "ColorSpaceTestSampleApp";
				const_cast< std::remove_const_t<decltype( m_window )>& >( m_window ) = m_winMgr->createWindow( std::move( params ) );
			}
			auto surface = CSurfaceVulkanWin32::create( smart_refctd_ptr( m_api ), smart_refctd_ptr_static_cast< ui::IWindowWin32 >( m_window ) );
			const_cast< std::remove_const_t<decltype( m_surface )>& >( m_surface ) = nbl::video::CSimpleResizeSurface<nbl::video::CDefaultSwapchainFramebuffers>::create( std::move( surface ) );
		}
		if( m_surface )
		{
			return { {m_surface->getSurface()/*,EQF_NONE*/} };
		}
		return {};
	}

private:
	smart_refctd_ptr<nbl::ui::IWindow> m_window;
	smart_refctd_ptr<CSimpleResizeSurface<CDefaultSwapchainFramebuffers>> m_surface;
	//
	smart_refctd_ptr<IGPUGraphicsPipeline> m_pipeline;
	// We can't use the same semaphore for acquire and present, because that would disable "Frames in Flight" by syncing previous present against next acquire.
	smart_refctd_ptr<ISemaphore> m_semaphore;
	// Use a separate counter to cycle through our resources for clarity
	uint64_t m_submitIx : 59 = 0;
	// Maximum frames which can be simultaneously rendered
	uint64_t m_maxFramesInFlight : 5;
	// Enough Command Buffers and other resources for all frames in flight!
	std::array<smart_refctd_ptr<IGPUDescriptorSet>, ISwapchain::MaxImages> m_descriptorSets;
	std::array<smart_refctd_ptr<IGPUCommandPool>, ISwapchain::MaxImages> m_cmdPools;
	std::array<smart_refctd_ptr<IGPUCommandBuffer>, ISwapchain::MaxImages> m_cmdBufs;

};


NBL_MAIN_FUNC( BoxBlurDemo )