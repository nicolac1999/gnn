from subsystems.heat_flux.utils.data_helpers import sample_generator
from subsystems.heat_flux.utils.plotting import plot_preview


frames = sample_generator(ds_filepath=r'data/heat/heat_dataset wet=2,t=1.8,K=3.45.xlsx',
                          variables=['T', 'WettedLength'],
                          starting_time=0, duration=None,
                          max_frames=3, samples_per_frame=31, time_step=0.5, stride=20)

plot_preview(ds_filepath=r'data/heat/heat_dataset wet=2,t=1.8,K=3.45.xlsx',
             variables=['T'],
             model_result_index=1,
             starting_time=0.,
             duration=100,
             time_step=0.5,
             frames=frames)