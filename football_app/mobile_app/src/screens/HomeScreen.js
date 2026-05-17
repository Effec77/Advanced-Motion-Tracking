
import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Alert
} from 'react-native';

const HomeScreen = ({ navigation }) => {
  const [recentAnalyses, setRecentAnalyses] = useState([]);
  
  const handleUploadVideo = () => {
    navigation.navigate('Upload');
  };
  
  const handleViewHistory = () => {
    navigation.navigate('History');
  };
  
  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Football Coach AI</Text>
        <Text style={styles.subtitle}>Improve your technique with AI analysis</Text>
      </View>
      
      <TouchableOpacity 
        style={styles.uploadButton}
        onPress={handleUploadVideo}
      >
        <Text style={styles.uploadButtonText}>Upload Video for Analysis</Text>
      </TouchableOpacity>
      
      <View style={styles.statsContainer}>
        <Text style={styles.sectionTitle}>Your Progress</Text>
        <View style={styles.statCard}>
          <Text style={styles.statValue}>7.8/10</Text>
          <Text style={styles.statLabel}>Average Technique Score</Text>
        </View>
      </View>
      
      <TouchableOpacity 
        style={styles.historyButton}
        onPress={handleViewHistory}
      >
        <Text style={styles.historyButtonText}>View Analysis History</Text>
      </TouchableOpacity>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    padding: 20,
    backgroundColor: '#2E8B57',
    alignItems: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 5,
  },
  subtitle: {
    fontSize: 16,
    color: 'white',
    opacity: 0.9,
  },
  uploadButton: {
    backgroundColor: '#FF6B35',
    margin: 20,
    padding: 20,
    borderRadius: 10,
    alignItems: 'center',
  },
  uploadButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  statsContainer: {
    margin: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 15,
    color: '#333',
  },
  statCard: {
    backgroundColor: 'white',
    padding: 20,
    borderRadius: 10,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  statValue: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#2E8B57',
  },
  statLabel: {
    fontSize: 16,
    color: '#666',
    marginTop: 5,
  },
  historyButton: {
    backgroundColor: 'white',
    margin: 20,
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#2E8B57',
  },
  historyButtonText: {
    color: '#2E8B57',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default HomeScreen;
