def promote_model(models):
    if not models:
        return None
    best_model = max(
        models,
        key=lambda m: (
            m["accuracy"],
            -m["latency"],
            m["timestamp"]
        )
    )
    return best_model["name"]