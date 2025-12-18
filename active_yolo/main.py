from config import DataConfig, TrainConfig, AppConfig


def main():
    print("Hello from ActiveYOLO!")

    data_config = DataConfig.load_data_config()
    print(f"Data configuration loaded: {data_config}")

    train_config = TrainConfig.load_train_config()
    print(f"Training configuration loaded: {train_config}")

    app_config = AppConfig.load_app_config()
    print(f"App configuration loaded: {app_config}")


if __name__ == "__main__":
    main()
