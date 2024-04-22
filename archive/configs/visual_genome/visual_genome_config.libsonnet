local vg_data_paths = {
  "image_data_path": "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/visual_genome/",
  "image_meta_file": "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/visual_genome/image_data.json",
  "region_description_file": "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/visual_genome/region_descriptions.json",
};

local vg_data_pipeline = {
  name: 'VisualGenomeDataPipeline',
  regenerate: false,
  do_inspect: true,
  transforms: {
    'input:LoadVisualGenomeData': {
      transform_name: 'LoadVisualGenomeData',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        data_paths: vg_data_paths,
      },
    },
    'process:PrepareVisualGenomeForRetrieval': {
      input_node: "input:LoadVisualGenomeData",
      transform_name: 'PrepareVisualGenomeForRetrieval',
      regenerate: false,
      cache: true,
      setup_kwargs: {
      },
    },
  },
};

{
  vg_data_pipeline: vg_data_pipeline,
}
