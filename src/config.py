class Config:
    BATCH_SIZE = 64
    BUFFER_SIZE = 20000
    EPOCHS = 20
    # To keep this example small and relatively fast, drop examples with a length of over 40 tokens.
    EXAMPLES_MAX_LENGTH = 40
    GPU_BATCH_SIZE = 50
    SHAPE = (64, 40)