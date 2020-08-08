# submit config

# test what happens when XY dont overlap and thus doesnt
# contribute to loss when it gets things wrong.
PYTHONPATH=.. python3 hydra_system.py
  -m model.z_size=9 \
  dataset=xyscale-no-overlap \
  framework=betavae,adavae,badavae \
  framework.cls.params.batch_logvar_mode='normal','reparameterize','all' \
