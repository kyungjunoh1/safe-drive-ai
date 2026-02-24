package com.safedriveai.backend.domain;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDateTime;

@Entity
@Table(name = "driving_records")
@Getter @Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class DrivingRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "driver_id", nullable = false)
    private Driver driver;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "inspection_log_id")
    private InspectionLog inspectionLog;

    @Column(nullable = false)
    private LocalDateTime startTime;

    private LocalDateTime endTime;

    @Column(nullable = false)
    private Integer drowsinessCount = 0;

    @Column(nullable = false)
    private Integer warningCount = 0;

    @Column(nullable = false)
    private Integer maxRiskLevel = 0;

    @Column(columnDefinition = "TEXT")
    private String notes;

    @PrePersist
    protected void onCreate() {
        this.startTime = LocalDateTime.now();
    }
}
