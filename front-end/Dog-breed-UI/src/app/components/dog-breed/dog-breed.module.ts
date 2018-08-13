import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { MatCardModule,MatButtonModule, MatProgressSpinnerModule} from '@angular/material';
import { DogBreedComponent } from './dog-breed.component';

@NgModule({
  declarations: [
    DogBreedComponent
  ],
  imports: [
    BrowserModule,
    MatCardModule,
    MatButtonModule,
    MatProgressSpinnerModule
  ],
  exports:[
    DogBreedComponent
  ],
  providers: []
})
export class DogBreedModule{ }
