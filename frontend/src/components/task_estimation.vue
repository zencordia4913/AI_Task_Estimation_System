<template>
  <div class="container mt-5">
    <div class="card shadow p-4">
      <h2 class="text-center text-primary">Task Duration Estimator</h2>

      <input
        v-model="taskName"
        type="text"
        placeholder="Enter task name..."
        class="form-control mt-3"
      />

      <p v-if="errorMessage" class="text-danger mt-2">{{ errorMessage }}</p>

      <h3 class="mt-4 text-secondary">Past Inferences</h3>
      <ul class="list-group mt-2">
        <transition-group name="fade">
          <li
            v-for="(entry, index) in estimatedDurations"
            :key="index"
            class="list-group-item d-flex flex-column"
          >
            <strong>{{ entry.task }}</strong>
            <span class="text-primary">Estimated: {{ entry.duration }} hrs</span>
            <small class="text-muted">Time: {{ entry.timeTaken }} ms</small>
            <small class="text-muted">Memory: {{ entry.memoryUsed }} MB</small>
          </li>
        </transition-group>
      </ul>
    </div>
  </div>
</template>

<script>
import axios from "axios";

export default {
  data() {
    return {
      taskName: "",
      estimatedDurations: [],
      isLoading: false,
      errorMessage: "",
    };
  },
  watch: {
    taskName(newTaskName) {
      if (newTaskName.trim().length > 0) {
        this.estimateDuration();
      }
    },
  },
  methods: {
    async estimateDuration() {
      if (!this.taskName.trim()) {
        this.errorMessage = "Task name cannot be empty!";
        return;
      }

      this.isLoading = true;
      this.errorMessage = "";

      const startTime = performance.now();

      try {
        const response = await axios.post("http://127.0.0.1:8000/api/predict/", {
          task_name: this.taskName,
        });

        const endTime = performance.now();
        const frontendInferenceTime = (endTime - startTime).toFixed(2);  // Time taken by Vue

        this.estimatedDurations.unshift({
          task: this.taskName,
          duration: response.data.estimated_duration,
          timeTaken: response.data.inference_time,
          memoryUsed: response.data.memory_used,
          frontendTime: frontendInferenceTime,
          timestamp: new Date().toLocaleTimeString(),
        });

      } catch (error) {
        console.error("API Error:", error.response?.data || error.message);
        this.errorMessage = "Error fetching data. Please try again.";
      } finally {
        this.isLoading = false;
      }
    },
  },
};
</script>
