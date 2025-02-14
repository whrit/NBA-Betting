import sqlite3
import time
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.regularizers import l1_l2

# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

class NBABettingModel:
    def __init__(self):
        self.scaler = StandardScaler()
        os.makedirs('../../Logs', exist_ok=True)
        os.makedirs('../../Models', exist_ok=True)
        
    def load_and_preprocess_data(self):
        try:
            dataset = "dataset_2012-25_new"
            con = sqlite3.connect("../../Data/dataset.sqlite")
            data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
            con.close()
            
            # Create feature engineering
            data['Home_Away_Win_PCT_Diff'] = data['W_PCT'] - data['W_PCT.1']
            data['Points_Diff_Per_Game'] = data['PTS'] / data['GP'] - data['PTS.1'] / data['GP.1']
            data['FG_PCT_Diff'] = data['FG_PCT'] - data['FG_PCT.1']
            data['FG3_PCT_Diff'] = data['FG3_PCT'] - data['FG3_PCT.1']
            data['Rest_Advantage'] = data['Days-Rest-Home'] - data['Days-Rest-Away']
            
            # Create rolling averages for key stats (last 10 games)
            rolling_columns = ['PTS', 'FG_PCT', 'FG3_PCT', 'REB', 'AST', 'TOV']
            for col in rolling_columns:
                data[f'{col}_Rolling_10'] = data.groupby('TEAM_NAME')[col].rolling(window=10).mean().reset_index(0, drop=True)
                data[f'{col}_Rolling_10_Away'] = data.groupby('TEAM_NAME.1')[f'{col}.1'].rolling(window=10).mean().reset_index(0, drop=True)
            
            # Store target variables
            self.scores = data['Score']
            self.margin = data['Home-Team-Win']
            self.ou = data['OU']
            self.ou_cover = data['OU-Cover']
            
            # Drop non-feature columns
            drop_columns = ['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 
                           'Date.1', 'OU', 'OU-Cover']
            data.drop(drop_columns, axis=1, inplace=True)
            
            # Handle missing values
            data = data.fillna(data.mean())
            
            # Scale features
            self.features = self.scaler.fit_transform(data)
            return self.features
            
        except Exception as e:
            print(f"Error: {str(e)}")
            raise
    
    def build_model(self, input_shape):
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(shape=input_shape),  # Changed from input_shape to shape
            
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1),
            
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train_model(self, epochs=100, batch_size=32):  # Changed batch size back to 32
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            self.features, self.margin, test_size=0.2, random_state=RANDOM_SEED,
            stratify=self.margin
        )
        
        # Make sure inputs are the right shape and type
        X_train = np.array(X_train, dtype=np.float32)
        X_val = np.array(X_val, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)
        y_val = np.array(y_val, dtype=np.int32)
        
        # Setup callbacks
        current_time = str(time.time())
        callbacks = [
            TensorBoard(log_dir=f'../../Logs/{current_time}'),
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ModelCheckpoint(
                filepath=f'../../Models/NBA-Model-{current_time}.keras',
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
        # Build and train model
        input_shape = (X_train.shape[1],)
        model = self.build_model(input_shape)
        
        # Print model summary and dataset shapes
        model.summary()
        print(f"\nTraining set shape: {X_train.shape}")
        print(f"Validation set shape: {X_val.shape}")
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            shuffle=True
        )
        
        return model, history
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance with various metrics"""
        # Ensure correct data types
        X_test = np.array(X_test, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.int32)
        
        predictions = model.predict(X_test, batch_size=32)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate various metrics
        accuracy = np.mean(predicted_classes == y_test)
        
        # Calculate ROI assuming equal betting units
        roi = self.calculate_roi(y_test, predicted_classes)
        
        return {
            'accuracy': accuracy,
            'roi': roi,
            'predictions': predictions,
            'predicted_classes': predicted_classes
        }
    
    @staticmethod
    def calculate_roi(actual, predicted, stake=100):
        """Calculate ROI based on predictions"""
        correct_bets = np.sum(actual == predicted)
        total_bets = len(actual)
        
        # Assuming -110 odds (1.91 decimal)
        winning_return = correct_bets * stake * 1.91
        total_cost = total_bets * stake
        
        roi = ((winning_return - total_cost) / total_cost) * 100
        return roi

# Usage example
if __name__ == "__main__":
    try:
        model = NBABettingModel()
        features = model.load_and_preprocess_data()
        
        # Print initial feature shape
        print(f"\nTotal dataset shape: {features.shape}")
        
        trained_model, history = model.train_model()
        
        # Split data for final evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            features, model.margin, test_size=0.1, random_state=RANDOM_SEED,
            stratify=model.margin
        )
        
        results = model.evaluate_model(trained_model, X_test, y_test)
        print(f"\nFinal Results:")
        print(f"Model Accuracy: {results['accuracy']:.2%}")
        print(f"ROI: {results['roi']:.2%}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")