package com.safedriveai.backend.domain;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDateTime;

@Entity
@Table(name = "alerts")
@Getter @Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Alert {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "driver_id", nullable = false)
    private Driver driver;

    @Column(nullable = false, length = 20)
    private String alertType;

    @Column(nullable = false)
    private LocalDateTime alertTime;

    @Column(length = 100)
    private String sentTo;

    @Column(nullable = false, length = 20)
    private String status = "SENT";

    @Column(columnDefinition = "TEXT")
    private String message;

    @PrePersist
    protected void onCreate(){
        this.alertTime = LocalDateTime.now();
    }
}
