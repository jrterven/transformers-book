for step in range(1, steps + 1):
    batch = collate(rng.sample(train, batch_size))
    training_nll = train_step(model, optimizer, batch)
    if step % validation_every == 0 or step == steps:
        metrics = evaluate(model, validation)
        history.append({"step": step, "training_batch_nll": training_nll,
                        "validation": metrics})
        print(f"step={step:4d} validation_nll={metrics['nll']:.5f} "
              f"exact_match={metrics['exact_match']:.3f}", flush=True)
        if metrics["nll"] < best_nll:
            best_nll = metrics["nll"]
            save_checkpoint(checkpoint_path, model, model_config,
                            experiment, step, metrics)
model, checkpoint = load_checkpoint(checkpoint_path)
test_metrics = evaluate(model, test)
