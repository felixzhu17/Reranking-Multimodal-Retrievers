// This is the configuration file for dataloaders. It registers what dataloaders are available to use
// For each dataloader, it also registers what dataset modules are available to obtain processed features
// All dataloader and feature loaders must be declared here for runway to work
local beans_data_pipeline = {
  name: 'BeansDataPipeline',
  regenerate: false,
  do_inspect: true,
  transforms: {
    'input:LoadBeansDataset': {
      transform_name: 'LoadBeansDataset',
      setup_kwargs: {
        dataset_name: 'beans-new',
      },
      regenerate: false,
      cache: true
    },
    'output:BeanJitterTransform': {
      input_node: 'input:LoadBeansDataset',
      transform_name: 'BeansJitterTransform',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        brightness: 0.5, 
        hue: 0.2
      },
    },
  }, 
};

{
  beans_data_pipeline: beans_data_pipeline,
}