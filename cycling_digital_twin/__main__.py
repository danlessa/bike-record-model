from .execution_logic import extrapolation_cycle
import click

@click.command()
@click.option('-p', '--past-days', 'past_days',
              default=14,
              help="Number of past days to download")
@click.option('-e', '--extrapolation-timesteps', 'extrapolation_timesteps',
              default=7 * 24,
              help="Number of extrapolation timesteps")
@click.option('-l', '--use-last-data', 'use_last_data',
              is_flag=True,
              help="Use last retrieved data rather than downloading it")
def main(use_last_data, past_days, extrapolation_timesteps) -> None:
    extrapolation_cycle(use_last_data=use_last_data,
                        historical_interval=past_days,
                        extrapolation_timesteps=extrapolation_timesteps)

    # %%


if __name__ == "__main__":
    main()