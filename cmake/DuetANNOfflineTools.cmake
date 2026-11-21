include_guard(GLOBAL)

function(duetann_add_offline_tools)
    add_executable(vamana_offline_builder
        tools/vamana_offline/vamana_offline_builder.cc
        ${DUETANN_OFFLINE_BUILDER_SOURCES}
    )
    duetann_target_tool_includes(vamana_offline_builder)
    target_link_libraries(vamana_offline_builder rdma_library)
    target_link_metis(vamana_offline_builder)

    add_executable(vamana_graph_repair
        tools/vamana_offline/vamana_graph_repair.cc
        tools/vamana_offline/checkpoint.cc
        tools/vamana_offline/dataset_io.cc
        tools/vamana_offline/graph.cc
        tools/vamana_offline/progress.cc
        tools/vamana_offline/recall_check.cc
        tools/vamana_offline/reverse_repair.cc
    )
    duetann_target_tool_includes(vamana_graph_repair)
    target_link_libraries(vamana_graph_repair rdma_library)

    add_executable(vamana_pq_indexer
        tools/vamana_offline/vamana_pq_indexer.cc
        ${DUETANN_PQ_INDEXER_SOURCES}
    )
    duetann_target_tool_includes(vamana_pq_indexer)
    target_link_libraries(vamana_pq_indexer rdma_library FaissCPU::FaissCPU)

    add_executable(vamana_graph_extent_indexer
        tools/vamana_offline/vamana_graph_extent_indexer.cc
        ${DUETANN_GRAPH_EXTENT_INDEXER_SOURCES}
    )
    duetann_target_tool_includes(vamana_graph_extent_indexer)
    target_link_libraries(vamana_graph_extent_indexer rdma_library)

    add_executable(vamana_legacy_index_converter
        tools/vamana_offline/vamana_legacy_index_converter.cc
        tools/vamana_offline/legacy_index_converter.cc
        ${DUETANN_PQ_INDEXER_SOURCES}
    )
    duetann_target_tool_includes(vamana_legacy_index_converter)
    target_link_libraries(vamana_legacy_index_converter
        rdma_library FaissCPU::FaissCPU)

    add_executable(vamana_metis_repartitioner
        tools/vamana_offline/vamana_metis_repartitioner.cc
        ${DUETANN_METIS_REPARTITIONER_SOURCES})
    duetann_target_tool_includes(vamana_metis_repartitioner)
    target_link_libraries(vamana_metis_repartitioner
        rdma_library FaissCPU::FaissCPU)
    target_link_metis(vamana_metis_repartitioner)
endfunction()
