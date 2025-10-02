# SLAB Project

## TODO list:

- [ ] Reenable plots in test_torch.
- [ ] Resolve mismatch of the outputs.
- [ ] Clean up type hints in methods in Contini.
- [ ] Add a webhook to automatically update the list of issues on GitHub.
- [ ] Source of the nans is in mode == "sum" of G_func()- ensure the addition is done correctly.
- [ ] Fix the source of nan in the argument of G_func().
- [ ] Try to move more methods and attributes into BaseClass and put it in a separate module.
- [ ] Add training.py to torch_modules/other.
- [ ] Make dictionary with functions or a function factory calling training loop functions.
- [ ] Clean up type hints in methods in tContini.
- [x] nan comes from the \*\* (3/4) of a negative value. Add a check or a filter for negative.
